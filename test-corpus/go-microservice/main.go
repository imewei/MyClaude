// Go HTTP microservice with goroutines, channels, and graceful shutdown.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

type WorkItem struct {
	ID      string `json:"id"`
	Payload string `json:"payload"`
}

type Result struct {
	ID     string `json:"id"`
	Status string `json:"status"`
}

func worker(id int, jobs <-chan WorkItem, results chan<- Result, wg *sync.WaitGroup) {
	defer wg.Done()
	for item := range jobs {
		log.Printf("worker %d processing job %s", id, item.ID)
		time.Sleep(100 * time.Millisecond) // simulate work
		results <- Result{ID: item.ID, Status: "completed"}
	}
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
}

func processHandler(jobs chan<- WorkItem) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var item WorkItem
		if err := json.NewDecoder(r.Body).Decode(&item); err != nil {
			http.Error(w, "invalid request body", http.StatusBadRequest)
			return
		}
		select {
		case jobs <- item:
			w.WriteHeader(http.StatusAccepted)
			json.NewEncoder(w).Encode(map[string]string{"status": "queued", "id": item.ID})
		default:
			http.Error(w, "queue full", http.StatusServiceUnavailable)
		}
	}
}

func main() {
	const numWorkers = 4
	jobs := make(chan WorkItem, 100)
	results := make(chan Result, 100)

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go worker(i, jobs, results, &wg)
	}

	go func() {
		for res := range results {
			log.Printf("completed: %s -> %s", res.ID, res.Status)
		}
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)
	mux.HandleFunc("/process", processHandler(jobs))

	srv := &http.Server{
		Addr:         fmt.Sprintf(":%s", getEnv("PORT", "8080")),
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	go func() {
		log.Printf("server starting on %s", srv.Addr)
		if err := srv.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("shutting down gracefully...")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	srv.Shutdown(ctx)
	close(jobs)
	wg.Wait()
	close(results)
}

func getEnv(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return fallback
}
