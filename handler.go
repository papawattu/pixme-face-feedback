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
	httpClient     *http.Client
	deepfaceClient *http.Client
}

func NewFeedbackHandler(pixmeAPIURL, imageBaseURL, deepfaceURL, facesDir, internalAPIKey string) *FeedbackHandler {
	return &FeedbackHandler{
		pixmeAPIURL:    pixmeAPIURL,
		imageBaseURL:   imageBaseURL,
		deepfaceURL:    deepfaceURL,
		facesDir:       facesDir,
		internalAPIKey: internalAPIKey,
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

// HandlePersonAdded is called when a person is manually tagged on an image.
// It downloads the image, detects faces via DeepFace /represent, crops the
// best face, and saves it to the faces directory for future recognition.
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

	// 3. Detect faces via DeepFace /represent.
	faces, err := detectFaces(ctx, h.deepfaceClient, h.deepfaceURL, imageURL)
	if err != nil {
		slog.Error("Failed to detect faces",
			slog.String("image_id", event.ImageID),
			slog.String("image_url", imageURL),
			slog.Any("error", err),
		)
		return
	}

	if len(faces) == 0 {
		slog.Warn("No faces detected in image, skipping",
			slog.String("image_id", event.ImageID),
		)
		return
	}

	// 4. Pick the best face (largest bounding box area).
	face := pickLargestFace(faces)
	slog.Info("Selected face for cropping",
		slog.String("image_id", event.ImageID),
		slog.Int("x", face.X),
		slog.Int("y", face.Y),
		slog.Int("w", face.W),
		slog.Int("h", face.H),
		slog.Int("total_faces", len(faces)),
	)

	// 5. Download the full image.
	fullImage, format, err := h.downloadImage(ctx, imageURL)
	if err != nil {
		slog.Error("Failed to download image",
			slog.String("image_id", event.ImageID),
			slog.String("image_url", imageURL),
			slog.Any("error", err),
		)
		return
	}

	// 6. Crop the face region (with padding).
	cropped := cropFace(fullImage, face, 0.2)

	// 7. Save to faces directory.
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
