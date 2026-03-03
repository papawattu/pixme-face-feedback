package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
)

// FacialArea is the bounding box returned by DeepFace /represent and /find.
type FacialArea struct {
	X int `json:"x"`
	Y int `json:"y"`
	W int `json:"w"`
	H int `json:"h"`
}

// RepresentResult is a single face entry from DeepFace /represent response.
type RepresentResult struct {
	FacialArea     FacialArea `json:"facial_area"`
	FaceConfidence float64    `json:"face_confidence"`
}

// RepresentResponse is the top-level response from DeepFace POST /represent.
type RepresentResponse struct {
	Results []RepresentResult `json:"results"`
}

// RepresentRequest is the request body for DeepFace POST /represent.
type RepresentRequest struct {
	Img              string `json:"img"`
	EnforceDetection bool   `json:"enforce_detection"`
	DetectorBackend  string `json:"detector_backend"`
}

// FindRequest is the request body for DeepFace POST /find.
type FindRequest struct {
	Img              string `json:"img"`
	DbPath           string `json:"db_path"`
	EnforceDetection bool   `json:"enforce_detection"`
	ModelName        string `json:"model_name"`
	DetectorBackend  string `json:"detector_backend"`
	DistanceMetric   string `json:"distance_metric"`
}

// FindResponse is the response from DeepFace POST /find.
// Keys are string indices ("0", "1", ...).
type FindResponse struct {
	Identity  map[string]string  `json:"identity"`
	Distance  map[string]float64 `json:"distance"`
	Threshold map[string]float64 `json:"threshold"`
	SourceX   map[string]float64 `json:"source_x"`
	SourceY   map[string]float64 `json:"source_y"`
	SourceW   map[string]float64 `json:"source_w"`
	SourceH   map[string]float64 `json:"source_h"`
}

// DeepFaceConfig holds configurable parameters for DeepFace requests.
type DeepFaceConfig struct {
	ModelName       string
	DetectorBackend string
	DistanceMetric  string
	FacesDir        string // path to the reference faces directory (e.g. /mnt/faces)
}

// detectFaces calls DeepFace POST /represent to detect faces in the given
// image URL and returns their bounding boxes.
func detectFaces(ctx context.Context, client *http.Client, deepfaceURL, imageURL string, cfg DeepFaceConfig) ([]FacialArea, error) {
	payload := RepresentRequest{
		Img:              imageURL,
		EnforceDetection: false,
		DetectorBackend:  cfg.DetectorBackend,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshaling represent request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, deepfaceURL+"/represent", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("calling DeepFace /represent: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		slog.Warn("DeepFace /represent non-200 response",
			slog.Int("status", resp.StatusCode),
			slog.String("body", string(respBody)),
		)
		return nil, fmt.Errorf("DeepFace /represent: %s — %s", resp.Status, string(respBody))
	}

	var representResp RepresentResponse
	if err := json.NewDecoder(resp.Body).Decode(&representResp); err != nil {
		return nil, fmt.Errorf("decoding /represent response: %w", err)
	}

	faces := make([]FacialArea, 0, len(representResp.Results))
	for _, r := range representResp.Results {
		faces = append(faces, r.FacialArea)
	}
	return faces, nil
}

// findPersonFace calls DeepFace POST /find to locate a specific person's face
// in an image using the reference face database. It returns the bounding box
// of the matched face, or nil if no match was found.
func findPersonFace(ctx context.Context, client *http.Client, deepfaceURL, imageURL, personName string, cfg DeepFaceConfig) (*FacialArea, error) {
	// Use the person-specific subdirectory as db_path so /find only matches
	// against that person's reference faces.
	personDbPath := cfg.FacesDir + "/" + personName

	payload := FindRequest{
		Img:              imageURL,
		DbPath:           personDbPath,
		EnforceDetection: false,
		ModelName:        cfg.ModelName,
		DetectorBackend:  cfg.DetectorBackend,
		DistanceMetric:   cfg.DistanceMetric,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshaling find request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, deepfaceURL+"/find", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("calling DeepFace /find: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("DeepFace /find: %s — %s", resp.Status, string(respBody))
	}

	var findResp FindResponse
	if err := json.NewDecoder(resp.Body).Decode(&findResp); err != nil {
		return nil, fmt.Errorf("decoding /find response: %w", err)
	}

	// Find the best (lowest distance) match that is within threshold
	// and belongs to the target person.
	var bestKey string
	bestDistance := -1.0
	for key, identity := range findResp.Identity {
		// Verify the match is actually for our target person.
		if !strings.Contains(identity, "/"+personName+"/") {
			continue
		}

		distance, hasDistance := findResp.Distance[key]
		threshold, hasThreshold := findResp.Threshold[key]

		// Skip matches that exceed the threshold.
		if hasDistance && hasThreshold && distance > threshold {
			continue
		}

		if bestDistance < 0 || distance < bestDistance {
			bestDistance = distance
			bestKey = key
		}
	}

	if bestKey == "" {
		return nil, nil // no match found
	}

	// Extract the source face bounding box from the match.
	x, hasX := findResp.SourceX[bestKey]
	y, hasY := findResp.SourceY[bestKey]
	w, hasW := findResp.SourceW[bestKey]
	h, hasH := findResp.SourceH[bestKey]

	if !hasX || !hasY || !hasW || !hasH {
		return nil, fmt.Errorf("matched identity but missing source bounding box for key %s", bestKey)
	}

	return &FacialArea{
		X: int(x),
		Y: int(y),
		W: int(w),
		H: int(h),
	}, nil
}

// pickLargestFace returns the face with the largest bounding box area.
// Assumes len(faces) > 0.
func pickLargestFace(faces []FacialArea) FacialArea {
	best := faces[0]
	bestArea := best.W * best.H
	for _, f := range faces[1:] {
		area := f.W * f.H
		if area > bestArea {
			best = f
			bestArea = area
		}
	}
	return best
}
