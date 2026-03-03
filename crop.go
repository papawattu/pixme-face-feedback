package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
)

// cropFace extracts a face region from the source image with padding.
// The padding factor (e.g. 0.2) expands each edge by that fraction of the
// face dimension, clamped to the image bounds.
func cropFace(src image.Image, face FacialArea, padding float64) image.Image {
	bounds := src.Bounds()

	// Calculate padded region.
	padW := int(float64(face.W) * padding)
	padH := int(float64(face.H) * padding)

	x0 := face.X - padW
	y0 := face.Y - padH
	x1 := face.X + face.W + padW
	y1 := face.Y + face.H + padH

	// Clamp to image bounds.
	if x0 < bounds.Min.X {
		x0 = bounds.Min.X
	}
	if y0 < bounds.Min.Y {
		y0 = bounds.Min.Y
	}
	if x1 > bounds.Max.X {
		x1 = bounds.Max.X
	}
	if y1 > bounds.Max.Y {
		y1 = bounds.Max.Y
	}

	rect := image.Rect(x0, y0, x1, y1)

	// Use SubImage if available (zero-copy for most image types).
	type subImager interface {
		SubImage(r image.Rectangle) image.Image
	}
	if si, ok := src.(subImager); ok {
		return si.SubImage(rect)
	}

	// Fallback: manual pixel copy into RGBA.
	dst := image.NewRGBA(image.Rect(0, 0, rect.Dx(), rect.Dy()))
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			dst.Set(x-rect.Min.X, y-rect.Min.Y, src.At(x, y))
		}
	}
	return dst
}

// saveCroppedFace saves a cropped face image to disk, creating parent
// directories as needed.
func saveCroppedFace(img image.Image, outputPath, format string) error {
	dir := filepath.Dir(outputPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("creating directory %s: %w", dir, err)
	}

	f, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("creating file %s: %w", outputPath, err)
	}
	defer f.Close()

	switch format {
	case "png":
		if err := png.Encode(f, img); err != nil {
			return fmt.Errorf("encoding PNG: %w", err)
		}
	default:
		// Default to JPEG (quality 95 for good face recognition quality).
		if err := jpeg.Encode(f, img, &jpeg.Options{Quality: 95}); err != nil {
			return fmt.Errorf("encoding JPEG: %w", err)
		}
	}
	return nil
}

// extensionForFormat returns the file extension for a given image format.
func extensionForFormat(format string) string {
	switch format {
	case "png":
		return ".png"
	case "gif":
		return ".gif"
	default:
		return ".jpg"
	}
}
