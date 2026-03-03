package main

import (
	"image"
	"image/color"
	"testing"
)

func TestCropFace_BasicCrop(t *testing.T) {
	// Create a 200x200 white image.
	img := image.NewRGBA(image.Rect(0, 0, 200, 200))
	for y := 0; y < 200; y++ {
		for x := 0; x < 200; x++ {
			img.Set(x, y, color.White)
		}
	}

	face := FacialArea{X: 50, Y: 50, W: 100, H: 100}
	cropped := cropFace(img, face, 0.0)

	bounds := cropped.Bounds()
	if bounds.Dx() != 100 || bounds.Dy() != 100 {
		t.Errorf("expected 100x100, got %dx%d", bounds.Dx(), bounds.Dy())
	}
}

func TestCropFace_WithPadding(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 200, 200))
	face := FacialArea{X: 50, Y: 50, W: 100, H: 100}
	cropped := cropFace(img, face, 0.2)

	bounds := cropped.Bounds()
	// 20% padding on each side: 100 + 2*20 = 140 per dimension
	if bounds.Dx() != 140 || bounds.Dy() != 140 {
		t.Errorf("expected 140x140, got %dx%d", bounds.Dx(), bounds.Dy())
	}
}

func TestCropFace_PaddingClampedToEdge(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	// Face near top-left corner, padding should be clamped.
	face := FacialArea{X: 5, Y: 5, W: 50, H: 50}
	cropped := cropFace(img, face, 0.5)

	bounds := cropped.Bounds()
	// Without clamping: x0=-20, y0=-20, x1=80, y1=80
	// Clamped: x0=0, y0=0, x1=80, y1=80 → 80x80
	if bounds.Dx() != 80 || bounds.Dy() != 80 {
		t.Errorf("expected 80x80, got %dx%d", bounds.Dx(), bounds.Dy())
	}
}

func TestPickLargestFace(t *testing.T) {
	faces := []FacialArea{
		{X: 0, Y: 0, W: 10, H: 10}, // area 100
		{X: 0, Y: 0, W: 50, H: 50}, // area 2500
		{X: 0, Y: 0, W: 20, H: 20}, // area 400
	}
	best := pickLargestFace(faces)
	if best.W != 50 || best.H != 50 {
		t.Errorf("expected 50x50 face, got %dx%d", best.W, best.H)
	}
}

func TestPickLargestFace_SingleFace(t *testing.T) {
	faces := []FacialArea{
		{X: 10, Y: 20, W: 30, H: 40},
	}
	best := pickLargestFace(faces)
	if best.X != 10 || best.Y != 20 || best.W != 30 || best.H != 40 {
		t.Errorf("expected {10,20,30,40}, got %+v", best)
	}
}

func TestExtensionForFormat(t *testing.T) {
	tests := []struct {
		format string
		want   string
	}{
		{"jpeg", ".jpg"},
		{"png", ".png"},
		{"gif", ".gif"},
		{"", ".jpg"},
		{"webp", ".jpg"},
	}
	for _, tt := range tests {
		got := extensionForFormat(tt.format)
		if got != tt.want {
			t.Errorf("extensionForFormat(%q) = %q, want %q", tt.format, got, tt.want)
		}
	}
}
