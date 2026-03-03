package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/nats-io/nats.go"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/trace"
)

var BuildVersion = "dev"

func getEnvWithDefault(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
}

// natsHeaderCarrier adapts nats.Header for OTel TextMapPropagator.
type natsHeaderCarrier nats.Header

func (c natsHeaderCarrier) Get(key string) string {
	return nats.Header(c).Get(key)
}

func (c natsHeaderCarrier) Set(key, value string) {
	nats.Header(c).Set(key, value)
}

func (c natsHeaderCarrier) Keys() []string {
	keys := make([]string, 0, len(c))
	for k := range c {
		keys = append(keys, k)
	}
	return keys
}

// PersonEvent is the JSON payload published by pixme-api for
// person_added_to_image and person_removed_from_image NATS subjects.
type PersonEvent struct {
	ImageID    string `json:"imageId"`
	PersonName string `json:"personName"`
}

func main() {
	slog.Info("Starting pixme-face-feedback", slog.String("version", BuildVersion))

	// Initialize OpenTelemetry tracing.
	ctx := context.Background()
	tp, err := initTracer(ctx, "pixme-face-feedback")
	if err != nil {
		slog.Warn("Failed to initialize OpenTelemetry tracer", slog.Any("error", err))
	} else {
		defer shutdownTracer(tp)
	}

	// Load configuration.
	natsURL := getEnvWithDefault("NATS_URL", "nats://nats.nats.svc.cluster.local:4222")
	pixmeAPIURL := getEnvWithDefault("PIXME_API_URL", "http://pixme-api-service.pixme.svc.cluster.local:8080")
	imageBaseURL := getEnvWithDefault("IMAGE_BASE_URL", "http://pixme-static.pixme.svc.cluster.local:80")
	deepfaceURL := getEnvWithDefault("DEEPFACE_URL", "http://deepface.pixme.svc.cluster.local:5000")
	facesDir := getEnvWithDefault("FACES_DIR", "/data/faces")
	internalAPIKey := getEnvWithDefault("INTERNAL_API_KEY", "")

	slog.Info("Configuration loaded",
		slog.String("nats_url", natsURL),
		slog.String("pixme_api_url", pixmeAPIURL),
		slog.String("image_base_url", imageBaseURL),
		slog.String("deepface_url", deepfaceURL),
		slog.String("faces_dir", facesDir),
	)

	// Create the feedback handler.
	handler := NewFeedbackHandler(pixmeAPIURL, imageBaseURL, deepfaceURL, facesDir, internalAPIKey)

	// Connect to NATS.
	nc, err := nats.Connect(natsURL)
	if err != nil {
		slog.Error("Failed to connect to NATS", slog.Any("error", err))
		os.Exit(1)
	}
	defer nc.Close()
	slog.Info("Connected to NATS", slog.String("url", natsURL))

	propagator := otel.GetTextMapPropagator()
	tracer := otel.Tracer("pixme-face-feedback")

	// Subscribe to person_added_to_image.
	subAdd, err := nc.Subscribe("person_added_to_image", func(msg *nats.Msg) {
		// Extract OTel trace context from NATS headers.
		msgCtx := propagator.Extract(context.Background(), natsHeaderCarrier(msg.Header))
		msgCtx, span := tracer.Start(msgCtx, "person_added_to_image receive",
			trace.WithSpanKind(trace.SpanKindConsumer),
		)
		defer span.End()

		var event PersonEvent
		if err := json.Unmarshal(msg.Data, &event); err != nil {
			slog.Error("Failed to unmarshal person_added_to_image event",
				slog.Any("error", err),
				slog.String("raw", string(msg.Data)),
			)
			return
		}

		slog.Info("Received person_added_to_image",
			slog.String("image_id", event.ImageID),
			slog.String("person_name", event.PersonName),
		)
		handler.HandlePersonAdded(msgCtx, event)
	})
	if err != nil {
		slog.Error("Failed to subscribe to person_added_to_image", slog.Any("error", err))
		os.Exit(1)
	}
	defer subAdd.Unsubscribe()

	// Subscribe to person_removed_from_image.
	subRemove, err := nc.Subscribe("person_removed_from_image", func(msg *nats.Msg) {
		msgCtx := propagator.Extract(context.Background(), natsHeaderCarrier(msg.Header))
		msgCtx, span := tracer.Start(msgCtx, "person_removed_from_image receive",
			trace.WithSpanKind(trace.SpanKindConsumer),
		)
		defer span.End()

		var event PersonEvent
		if err := json.Unmarshal(msg.Data, &event); err != nil {
			slog.Error("Failed to unmarshal person_removed_from_image event",
				slog.Any("error", err),
				slog.String("raw", string(msg.Data)),
			)
			return
		}

		slog.Info("Received person_removed_from_image",
			slog.String("image_id", event.ImageID),
			slog.String("person_name", event.PersonName),
		)
		handler.HandlePersonRemoved(msgCtx, event)
	})
	if err != nil {
		slog.Error("Failed to subscribe to person_removed_from_image", slog.Any("error", err))
		os.Exit(1)
	}
	defer subRemove.Unsubscribe()

	// Health endpoint (for k8s liveness/readiness probes).
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		if !nc.IsConnected() {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprint(w, "NATS disconnected")
			return
		}
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "ok")
	})

	server := &http.Server{
		Addr:              ":8080",
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		slog.Info("Health endpoint listening", slog.String("addr", ":8080"))
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("Health server error", slog.Any("error", err))
		}
	}()

	// Wait for shutdown signal.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	sig := <-sigCh
	slog.Info("Received signal, shutting down", slog.String("signal", sig.String()))

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		slog.Error("Health server shutdown error", slog.Any("error", err))
	}

	// Drain NATS subscriptions (finish in-flight messages).
	if err := nc.Drain(); err != nil {
		slog.Error("NATS drain error", slog.Any("error", err))
	}

	slog.Info("Shutdown complete")
}
