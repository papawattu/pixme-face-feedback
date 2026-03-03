package main

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"time"

	"github.com/nats-io/nats.go"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
)

var handlerTracer = otel.Tracer("pixme-face-feedback/handler")

// FeedbackHandler processes person add/remove events by cropping faces
// from images and saving/removing them from the shared face database.
type FeedbackHandler struct {
	pixmeAPIURL    string
	imageBaseURL   string
	deepfaceURL    string
	facesDir       string
	internalAPIKey string
	deepfaceCfg    DeepFaceConfig
	httpClient     *http.Client
	deepfaceClient *http.Client
	natsConn       *nats.Conn // for publishing face_reference_added events
}

func NewFeedbackHandler(pixmeAPIURL, imageBaseURL, deepfaceURL, facesDir, internalAPIKey string, deepfaceCfg DeepFaceConfig, nc *nats.Conn) *FeedbackHandler {
	return &FeedbackHandler{
		pixmeAPIURL:    pixmeAPIURL,
		imageBaseURL:   imageBaseURL,
		deepfaceURL:    deepfaceURL,
		facesDir:       facesDir,
		internalAPIKey: internalAPIKey,
		deepfaceCfg:    deepfaceCfg,
		natsConn:       nc,
		httpClient: &http.Client{
			Timeout:   30 * time.Second,
			Transport: otelhttp.NewTransport(http.DefaultTransport),
		},
		deepfaceClient: &http.Client{
			Timeout:   120 * time.Second,
			Transport: otelhttp.NewTransport(http.DefaultTransport),
		},
	}
}

// ImageResponse is the wrapper returned by GET /api/images/{id}.
type ImageResponse struct {
	Image ImageDescriptor `json:"image"`
}

type ImageDescriptor struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	URI          string   `json:"uri"`
	ThumbnailURI string   `json:"thumbnailUri"`
	Title        string   `json:"title"`
	Description  string   `json:"description"`
	Categories   []string `json:"categories"`
	People       []string `json:"people"`
}

// FaceReferenceEvent is published to NATS when a new reference face is saved,
// allowing downstream services to trigger re-recognition.
type FaceReferenceEvent struct {
	PersonName string `json:"personName"`
	ImageID    string `json:"imageId"`
	FaceFile   string `json:"faceFile"`
}

// HandlePersonAdded is called when a person is tagged on an image.
// It locates the correct face using DeepFace /find (when reference faces
// exist) or falls back to detecting all faces and picking the largest one.
// The cropped face is saved to the faces directory for future recognition.
func (h *FeedbackHandler) HandlePersonAdded(ctx context.Context, event PersonEvent) {
	ctx, span := handlerTracer.Start(ctx, "HandlePersonAdded")
	defer span.End()
	span.SetAttributes(
		attribute.String("image_id", event.ImageID),
		attribute.String("person_name", event.PersonName),
	)

	// 1. Fetch image metadata from pixme-api.
	img, err := h.fetchImageMetadata(ctx, event.ImageID)
	if err != nil {
		slog.Error("Failed to fetch image metadata",
			slog.String("image_id", event.ImageID),
			slog.Any("error", err),
		)
		return
	}

	// 2. Build the full image URL.
	imageURL, err := h.buildImageURL(img.URI)
	if err != nil {
		slog.Error("Failed to build image URL",
			slog.String("image_id", event.ImageID),
			slog.Any("error", err),
		)
		return
	}

	// 3. Locate the correct face to crop.
	face, err := h.locateFace(ctx, imageURL, event)
	if err != nil {
		slog.Error("Failed to locate face",
			slog.String("image_id", event.ImageID),
			slog.String("person_name", event.PersonName),
			slog.Any("error", err),
		)
		return
	}
	if face == nil {
		slog.Warn("No face found in image, skipping",
			slog.String("image_id", event.ImageID),
		)
		return
	}

	slog.Info("Selected face for cropping",
		slog.String("image_id", event.ImageID),
		slog.String("person_name", event.PersonName),
		slog.Int("x", face.X),
		slog.Int("y", face.Y),
		slog.Int("w", face.W),
		slog.Int("h", face.H),
	)

	// 4. Download the full image.
	fullImage, format, err := h.downloadImage(ctx, imageURL)
	if err != nil {
		slog.Error("Failed to download image",
			slog.String("image_id", event.ImageID),
			slog.String("image_url", imageURL),
			slog.Any("error", err),
		)
		return
	}

	// 5. Crop the face region (with padding).
	cropped := cropFace(fullImage, *face, 0.2)

	// 6. Save to faces directory.
	outputDir := filepath.Join(h.facesDir, event.PersonName)
	outputFile := filepath.Join(outputDir, event.ImageID+extensionForFormat(format))

	if err := saveCroppedFace(cropped, outputFile, format); err != nil {
		slog.Error("Failed to save cropped face",
			slog.String("output_file", outputFile),
			slog.Any("error", err),
		)
		return
	}

	slog.Info("Successfully saved reference face",
		slog.String("person_name", event.PersonName),
		slog.String("image_id", event.ImageID),
		slog.String("output_file", outputFile),
	)

	// 7. Publish face_reference_added event for potential re-recognition.
	h.publishFaceReferenceAdded(ctx, event, outputFile)
}

// locateFace determines the correct face bounding box to crop.
// Strategy:
//  1. If the person already has reference faces, use DeepFace /find to match
//     the correct face in the image (handles multi-person photos accurately).
//  2. Otherwise, fall back to detecting all faces via /represent and picking
//     the largest one.
func (h *FeedbackHandler) locateFace(ctx context.Context, imageURL string, event PersonEvent) (*FacialArea, error) {
	// Check if this person already has reference faces.
	if h.hasReferenceFaces(event.PersonName) {
		slog.Info("Person has reference faces, using /find to locate correct face",
			slog.String("person_name", event.PersonName),
		)

		face, err := findPersonFace(ctx, h.deepfaceClient, h.deepfaceURL, imageURL, event.PersonName, h.deepfaceCfg)
		if err != nil {
			slog.Warn("DeepFace /find failed, falling back to /represent",
				slog.String("person_name", event.PersonName),
				slog.Any("error", err),
			)
			// Fall through to the detect+pick fallback below.
		} else if face != nil {
			slog.Info("Found matching face via /find",
				slog.String("person_name", event.PersonName),
				slog.Int("x", face.X),
				slog.Int("y", face.Y),
				slog.Int("w", face.W),
				slog.Int("h", face.H),
			)
			return face, nil
		} else {
			slog.Info("No match found via /find, falling back to /represent",
				slog.String("person_name", event.PersonName),
			)
		}
	} else {
		slog.Info("No reference faces for person, using /represent to detect faces",
			slog.String("person_name", event.PersonName),
		)
	}

	// Fallback: detect all faces and pick the largest.
	faces, err := detectFaces(ctx, h.deepfaceClient, h.deepfaceURL, imageURL, h.deepfaceCfg)
	if err != nil {
		return nil, fmt.Errorf("detecting faces: %w", err)
	}

	if len(faces) == 0 {
		return nil, nil
	}

	largest := pickLargestFace(faces)
	slog.Info("Picked largest face from detection",
		slog.Int("total_faces", len(faces)),
	)
	return &largest, nil
}

// hasReferenceFaces checks if the person has any existing reference face
// images in the faces directory.
func (h *FeedbackHandler) hasReferenceFaces(personName string) bool {
	personDir := filepath.Join(h.facesDir, personName)
	entries, err := os.ReadDir(personDir)
	if err != nil {
		return false // directory doesn't exist or can't be read
	}
	return len(entries) > 0
}

// publishFaceReferenceAdded publishes a face_reference_added NATS event
// so downstream services can trigger re-recognition if needed.
func (h *FeedbackHandler) publishFaceReferenceAdded(ctx context.Context, event PersonEvent, faceFile string) {
	if h.natsConn == nil {
		return
	}

	refEvent := FaceReferenceEvent{
		PersonName: event.PersonName,
		ImageID:    event.ImageID,
		FaceFile:   faceFile,
	}

	data, err := json.Marshal(refEvent)
	if err != nil {
		slog.Error("Failed to marshal face_reference_added event",
			slog.Any("error", err),
		)
		return
	}

	// Inject OTel trace context into NATS headers.
	msg := &nats.Msg{
		Subject: "face_reference_added",
		Data:    data,
		Header:  nats.Header{},
	}
	propagator := otel.GetTextMapPropagator()
	propagator.Inject(ctx, natsHeaderCarrier(msg.Header))

	if err := h.natsConn.PublishMsg(msg); err != nil {
		slog.Error("Failed to publish face_reference_added event",
			slog.String("person_name", event.PersonName),
			slog.String("image_id", event.ImageID),
			slog.Any("error", err),
		)
	} else {
		slog.Info("Published face_reference_added event",
			slog.String("person_name", event.PersonName),
			slog.String("image_id", event.ImageID),
		)
	}
}

// HandlePersonRemoved deletes the reference face image for the given person
// and image ID. If the person's directory becomes empty, it is also removed.
func (h *FeedbackHandler) HandlePersonRemoved(ctx context.Context, event PersonEvent) {
	ctx, span := handlerTracer.Start(ctx, "HandlePersonRemoved")
	defer span.End()
	span.SetAttributes(
		attribute.String("image_id", event.ImageID),
		attribute.String("person_name", event.PersonName),
	)

	personDir := filepath.Join(h.facesDir, event.PersonName)

	// Try common extensions.
	extensions := []string{".jpg", ".jpeg", ".png"}
	removed := false
	for _, ext := range extensions {
		filePath := filepath.Join(personDir, event.ImageID+ext)
		if err := os.Remove(filePath); err == nil {
			slog.Info("Removed reference face",
				slog.String("person_name", event.PersonName),
				slog.String("image_id", event.ImageID),
				slog.String("file", filePath),
			)
			removed = true
			break
		}
	}

	if !removed {
		slog.Warn("No reference face file found to remove",
			slog.String("person_name", event.PersonName),
			slog.String("image_id", event.ImageID),
			slog.String("dir", personDir),
		)
		return
	}

	// Clean up empty directory.
	entries, err := os.ReadDir(personDir)
	if err == nil && len(entries) == 0 {
		if err := os.Remove(personDir); err != nil {
			slog.Warn("Failed to remove empty person directory",
				slog.String("dir", personDir),
				slog.Any("error", err),
			)
		} else {
			slog.Info("Removed empty person directory",
				slog.String("person_name", event.PersonName),
				slog.String("dir", personDir),
			)
		}
	}
}

// fetchImageMetadata calls GET /api/images/{id} on pixme-api.
func (h *FeedbackHandler) fetchImageMetadata(ctx context.Context, imageID string) (*ImageDescriptor, error) {
	reqURL := fmt.Sprintf("%s/api/images/%s", h.pixmeAPIURL, url.PathEscape(imageID))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, err
	}
	if h.internalAPIKey != "" {
		req.Header.Set("X-Internal-Key", h.internalAPIKey)
	}

	resp, err := h.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("GET %s: %w", reqURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("GET %s: %s — %s", reqURL, resp.Status, string(body))
	}

	var imageResp ImageResponse
	if err := json.NewDecoder(resp.Body).Decode(&imageResp); err != nil {
		return nil, fmt.Errorf("decoding image response: %w", err)
	}
	return &imageResp.Image, nil
}

// buildImageURL constructs the full URL for downloading an image.
func (h *FeedbackHandler) buildImageURL(uri string) (string, error) {
	decoded, err := url.PathUnescape(uri)
	if err != nil {
		return "", fmt.Errorf("decoding URI %q: %w", uri, err)
	}
	return h.imageBaseURL + decoded, nil
}

// downloadImage fetches and decodes an image from the given URL.
func (h *FeedbackHandler) downloadImage(ctx context.Context, imageURL string) (image.Image, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, imageURL, nil)
	if err != nil {
		return nil, "", err
	}

	resp, err := h.httpClient.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("downloading image: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("downloading image: %s", resp.Status)
	}

	img, format, err := image.Decode(resp.Body)
	if err != nil {
		return nil, "", fmt.Errorf("decoding image: %w", err)
	}
	return img, format, nil
}
