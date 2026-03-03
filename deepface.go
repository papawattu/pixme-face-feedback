package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
)

// FacialArea is the bounding box returned by DeepFace /represent.
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
}

// detectFaces calls DeepFace POST /represent to detect faces in the given
// image URL and returns their bounding boxes.
func detectFaces(ctx context.Context, client *http.Client, deepfaceURL, imageURL string) ([]FacialArea, error) {
	payload := RepresentRequest{
		Img:              imageURL,
		EnforceDetection: false,
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
