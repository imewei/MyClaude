# Distributed Tracing Guide

Comprehensive guide to implementing distributed tracing with OpenTelemetry, Jaeger, and Tempo for microservices observability.

## Table of Contents

1. [Distributed Tracing Fundamentals](#distributed-tracing-fundamentals)
2. [OpenTelemetry SDK Setup](#opentelemetry-sdk-setup)
3. [Trace Context Propagation](#trace-context-propagation)
4. [Jaeger Deployment](#jaeger-deployment)
5. [Tempo Integration](#tempo-integration)
6. [Trace Sampling Strategies](#trace-sampling-strategies)
7. [Span Attributes and Semantics](#span-attributes-and-semantics)
8. [Trace Correlation](#trace-correlation)
9. [Service Mesh Tracing](#service-mesh-tracing)
10. [Performance Optimization](#performance-optimization)

## Distributed Tracing Fundamentals

### Core Concepts

**Trace**: End-to-end path of a request through distributed systems
**Span**: Single unit of work within a trace (operation, RPC call, database query)
**Trace Context**: Metadata propagated across service boundaries (trace ID, span ID, flags)
**Parent-Child Relationships**: Hierarchical structure of spans showing causality

### Terminology

```text
Trace ID: Unique identifier for entire request flow (128-bit)
Span ID: Unique identifier for individual operation (64-bit)
Parent Span ID: Links child span to parent
Trace State: Vendor-specific data propagated with trace
Sampling: Decision to record and export trace data
```

### Trace Structure Example

```text
Trace: order-processing (trace_id: 4bf92f3577b34da6a3ce929d0e0e4736)
├── Span: api-gateway.handle-request (span_id: 00f067aa0ba902b7)
│   ├── Span: auth-service.validate-token (span_id: 1f2e3d4c5b6a7988)
│   ├── Span: order-service.create-order (span_id: 2e3f4d5c6b7a8899)
│   │   ├── Span: db.insert-order (span_id: 3f4e5d6c7b8a9900)
│   │   └── Span: inventory-service.reserve (span_id: 4e5f6d7c8b9a0011)
│   └── Span: notification-service.send-email (span_id: 5f6e7d8c9b0a1122)
```

### Benefits

- **Root Cause Analysis**: Quickly identify failing services
- **Performance Profiling**: Find bottlenecks across services
- **Dependency Mapping**: Understand service interactions
- **SLA Monitoring**: Track request latencies end-to-end
- **Distributed Context**: Maintain request context across boundaries

## OpenTelemetry SDK Setup

### Node.js/TypeScript

```typescript
// tracing.ts
import { NodeSDK } from '@opentelemetry/sdk-node';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';
import { BatchSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { ParentBasedSampler, TraceIdRatioBasedSampler } from '@opentelemetry/sdk-trace-node';
import { W3CTraceContextPropagator } from '@opentelemetry/core';
import { B3Propagator, B3InjectEncoding } from '@opentelemetry/propagator-b3';
import { JaegerPropagator } from '@opentelemetry/propagator-jaeger';
import { CompositePropagator } from '@opentelemetry/core';

export function initializeTracing(serviceName: string): NodeSDK {
  // Configure resource with service metadata
  const resource = new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: serviceName,
    [SemanticResourceAttributes.SERVICE_VERSION]: process.env.SERVICE_VERSION || '1.0.0',
    [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: process.env.NODE_ENV || 'development',
    'service.instance.id': process.env.HOSTNAME || 'localhost',
  });

  // Configure OTLP exporter
  const traceExporter = new OTLPTraceExporter({
    url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://localhost:4318/v1/traces',
    headers: {
      'Authorization': `Bearer ${process.env.OTEL_AUTH_TOKEN || ''}`,
    },
  });

  // Configure sampling strategy
  const sampler = new ParentBasedSampler({
    root: new TraceIdRatioBasedSampler(parseFloat(process.env.TRACE_SAMPLE_RATE || '0.1')),
  });

  // Configure context propagation
  const propagator = new CompositePropagator({
    propagators: [
      new W3CTraceContextPropagator(),
      new B3Propagator({ injectEncoding: B3InjectEncoding.MULTI_HEADER }),
      new JaegerPropagator(),
    ],
  });

  // Initialize SDK
  const sdk = new NodeSDK({
    resource,
    traceExporter,
    spanProcessor: new BatchSpanProcessor(traceExporter, {
      maxQueueSize: 2048,
      maxExportBatchSize: 512,
      scheduledDelayMillis: 5000,
      exportTimeoutMillis: 30000,
    }),
    sampler,
    instrumentations: [
      getNodeAutoInstrumentations({
        '@opentelemetry/instrumentation-fs': { enabled: false },
        '@opentelemetry/instrumentation-http': {
          ignoreIncomingPaths: ['/health', '/metrics'],
          headersToSpanAttributes: {
            client: ['user-agent', 'x-request-id'],
            server: ['content-type'],
          },
        },
        '@opentelemetry/instrumentation-express': {
          requestHook: (span, requestInfo) => {
            span.setAttribute('http.route', requestInfo.route || 'unknown');
            span.setAttribute('http.user_agent', requestInfo.request.get('user-agent') || '');
          },
        },
      }),
    ],
    textMapPropagator: propagator,
  });

  sdk.start();

  // Graceful shutdown
  process.on('SIGTERM', () => {
    sdk.shutdown()
      .then(() => console.log('Tracing terminated'))
      .catch((error) => console.error('Error terminating tracing', error))
      .finally(() => process.exit(0));
  });

  return sdk;
}

// Manual instrumentation example
import { trace, context, SpanStatusCode } from '@opentelemetry/api';

export class OrderService {
  private tracer = trace.getTracer('order-service', '1.0.0');

  async createOrder(userId: string, items: OrderItem[]): Promise<Order> {
    return await this.tracer.startActiveSpan('createOrder', async (span) => {
      try {
        span.setAttributes({
          'user.id': userId,
          'order.items.count': items.length,
          'order.total': this.calculateTotal(items),
        });

        // Validate inventory
        const inventorySpan = this.tracer.startSpan('validateInventory', {
          parent: span,
          attributes: { 'inventory.items': items.length },
        });

        await this.validateInventory(items);
        inventorySpan.setStatus({ code: SpanStatusCode.OK });
        inventorySpan.end();

        // Create order in database
        const dbSpan = this.tracer.startSpan('database.insertOrder', {
          parent: span,
          attributes: {
            'db.system': 'postgresql',
            'db.operation': 'INSERT',
            'db.table': 'orders',
          },
        });

        const order = await this.db.insertOrder(userId, items);
        dbSpan.setAttribute('order.id', order.id);
        dbSpan.end();

        // Add span event
        span.addEvent('order_created', {
          'order.id': order.id,
          'processing.time.ms': Date.now() - span.startTime[0],
        });

        span.setStatus({ code: SpanStatusCode.OK });
        return order;
      } catch (error) {
        span.recordException(error as Error);
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: (error as Error).message,
        });
        throw error;
      } finally {
        span.end();
      }
    });
  }

  // Propagate context to external service
  async notifyShipping(orderId: string): Promise<void> {
    const activeContext = context.active();
    const span = trace.getSpan(activeContext);

    if (span) {
      const headers: Record<string, string> = {};
      trace.getTracerProvider().getDelegate()
        .getActiveSpanProcessor()
        .getTraceContext()
        .inject(activeContext, headers);

      await fetch('http://shipping-service/notify', {
        method: 'POST',
        headers: {
          ...headers,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ orderId }),
      });
    }
  }
}
```

### Python

```python
# tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, DEPLOYMENT_ENVIRONMENT
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatioBased
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.propagators.jaeger import JaegerPropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
import os
import logging

logger = logging.getLogger(__name__)

def initialize_tracing(service_name: str) -> None:
    """Initialize OpenTelemetry tracing with OTLP export."""

    # Configure resource
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: os.getenv('SERVICE_VERSION', '1.0.0'),
        DEPLOYMENT_ENVIRONMENT: os.getenv('ENVIRONMENT', 'development'),
        'service.instance.id': os.getenv('HOSTNAME', 'localhost'),
    })

    # Configure sampler
    sampler = ParentBasedTraceIdRatioBased(
        float(os.getenv('TRACE_SAMPLE_RATE', '0.1'))
    )

    # Create tracer provider
    provider = TracerProvider(
        resource=resource,
        sampler=sampler,
    )

    # Configure OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318/v1/traces'),
        headers={'Authorization': f"Bearer {os.getenv('OTEL_AUTH_TOKEN', '')}"},
    )

    # Add batch span processor
    provider.add_span_processor(
        BatchSpanProcessor(
            otlp_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=5000,
            export_timeout_millis=30000,
        )
    )

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    # Configure composite propagator
    set_global_textmap(
        CompositePropagator([
            TraceContextTextMapPropagator(),
            B3MultiFormat(),
            JaegerPropagator(),
        ])
    )

    logger.info(f"Tracing initialized for service: {service_name}")

def instrument_flask_app(app):
    """Instrument Flask application."""
    FlaskInstrumentor().instrument_app(
        app,
        excluded_urls="/health,/metrics",
        request_hook=request_hook,
        response_hook=response_hook,
    )
    RequestsInstrumentor().instrument()

def request_hook(span, environ):
    """Add custom attributes to request spans."""
    span.set_attribute('http.route', environ.get('PATH_INFO', 'unknown'))
    span.set_attribute('http.user_agent', environ.get('HTTP_USER_AGENT', ''))
    span.set_attribute('http.client_ip', environ.get('REMOTE_ADDR', ''))

def response_hook(span, status_code, response_headers):
    """Add response attributes to spans."""
    span.set_attribute('http.status_code', status_code)

# Manual instrumentation example
from opentelemetry.trace import Status, StatusCode
from typing import List, Dict
import time

class PaymentProcessor:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__, '1.0.0')

    def process_payment(self, order_id: str, amount: float, payment_method: str) -> Dict:
        """Process payment with distributed tracing."""
        with self.tracer.start_as_current_span('process_payment') as span:
            try:
                # Add span attributes
                span.set_attributes({
                    'payment.order_id': order_id,
                    'payment.amount': amount,
                    'payment.method': payment_method,
                    'payment.currency': 'USD',
                })

                # Validate payment
                with self.tracer.start_as_current_span('validate_payment') as validate_span:
                    validate_span.set_attribute('validation.type', 'fraud_check')
                    self.validate_payment(payment_method, amount)

                # Charge payment gateway
                with self.tracer.start_as_current_span('charge_gateway') as gateway_span:
                    gateway_span.set_attributes({
                        'gateway.provider': 'stripe',
                        'gateway.operation': 'charge',
                    })

                    start_time = time.time()
                    transaction_id = self.charge_stripe(amount, payment_method)
                    duration_ms = (time.time() - start_time) * 1000

                    gateway_span.set_attribute('gateway.transaction_id', transaction_id)
                    gateway_span.set_attribute('gateway.duration_ms', duration_ms)

                # Add event
                span.add_event('payment_completed', {
                    'transaction.id': transaction_id,
                    'processing.time_ms': duration_ms,
                })

                span.set_status(Status(StatusCode.OK))
                return {
                    'success': True,
                    'transaction_id': transaction_id,
                    'order_id': order_id,
                }

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def charge_stripe(self, amount: float, payment_method: str) -> str:
        """Simulate Stripe API call."""
        # Implementation with automatic HTTP instrumentation
        import requests
        response = requests.post(
            'https://api.stripe.com/v1/charges',
            data={'amount': int(amount * 100), 'currency': 'usd'},
        )
        return response.json()['id']

    def validate_payment(self, payment_method: str, amount: float) -> None:
        """Validate payment details."""
        if amount <= 0:
            raise ValueError("Invalid payment amount")
```

### Go

```go
// tracing.go
package tracing

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"go.opentelemetry.io/contrib/propagators/b3"
	"go.opentelemetry.io/contrib/propagators/jaeger"
)

// InitTracing initializes OpenTelemetry tracing
func InitTracing(serviceName, serviceVersion string) (*sdktrace.TracerProvider, error) {
	ctx := context.Background()

	// Create resource
	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName(serviceName),
			semconv.ServiceVersion(serviceVersion),
			semconv.DeploymentEnvironment(getEnv("ENVIRONMENT", "development")),
			attribute.String("service.instance.id", getEnv("HOSTNAME", "localhost")),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Create OTLP exporter
	exporter, err := otlptracehttp.New(ctx,
		otlptracehttp.WithEndpoint(getEnv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4318")),
		otlptracehttp.WithInsecure(),
		otlptracehttp.WithHeaders(map[string]string{
			"Authorization": fmt.Sprintf("Bearer %s", getEnv("OTEL_AUTH_TOKEN", "")),
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create exporter: %w", err)
	}

	// Configure sampler
	sampleRate := 0.1
	sampler := sdktrace.ParentBased(
		sdktrace.TraceIDRatioBased(sampleRate),
	)

	// Create tracer provider
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSampler(sampler),
		sdktrace.WithResource(res),
		sdktrace.WithBatcher(exporter,
			sdktrace.WithMaxQueueSize(2048),
			sdktrace.WithMaxExportBatchSize(512),
			sdktrace.WithBatchTimeout(5*time.Second),
			sdktrace.WithExportTimeout(30*time.Second),
		),
	)

	// Set global tracer provider
	otel.SetTracerProvider(tp)

	// Configure composite propagator
	otel.SetTextMapPropagator(
		propagation.NewCompositeTextMapPropagator(
			propagation.TraceContext{},
			propagation.Baggage{},
			b3.New(),
			jaeger.Jaeger{},
		),
	)

	log.Printf("Tracing initialized for service: %s", serviceName)
	return tp, nil
}

// Shutdown gracefully shuts down tracing
func Shutdown(ctx context.Context, tp *sdktrace.TracerProvider) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	if err := tp.Shutdown(ctx); err != nil {
		return fmt.Errorf("failed to shutdown tracer provider: %w", err)
	}
	return nil
}

// OrderService demonstrates manual instrumentation
type OrderService struct {
	tracer trace.Tracer
}

func NewOrderService() *OrderService {
	return &OrderService{
		tracer: otel.Tracer("order-service", trace.WithInstrumentationVersion("1.0.0")),
	}
}

func (s *OrderService) CreateOrder(ctx context.Context, userID string, items []OrderItem) (*Order, error) {
	ctx, span := s.tracer.Start(ctx, "CreateOrder",
		trace.WithSpanKind(trace.SpanKindServer),
		trace.WithAttributes(
			attribute.String("user.id", userID),
			attribute.Int("order.items.count", len(items)),
		),
	)
	defer span.End()

	// Validate inventory
	if err := s.validateInventory(ctx, items); err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "inventory validation failed")
		return nil, err
	}

	// Insert order into database
	order, err := s.insertOrder(ctx, userID, items)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "failed to insert order")
		return nil, err
	}

	// Add span attributes and events
	span.SetAttributes(
		attribute.String("order.id", order.ID),
		attribute.Float64("order.total", order.Total),
	)
	span.AddEvent("order_created", trace.WithAttributes(
		attribute.String("order.id", order.ID),
		attribute.String("order.status", "pending"),
	))

	span.SetStatus(codes.Ok, "order created successfully")
	return order, nil
}

func (s *OrderService) validateInventory(ctx context.Context, items []OrderItem) error {
	_, span := s.tracer.Start(ctx, "validateInventory",
		trace.WithAttributes(
			attribute.Int("inventory.items", len(items)),
		),
	)
	defer span.End()

	// Validation logic
	for _, item := range items {
		if item.Quantity <= 0 {
			return fmt.Errorf("invalid quantity for item: %s", item.ProductID)
		}
	}

	span.SetStatus(codes.Ok, "inventory validated")
	return nil
}

func (s *OrderService) insertOrder(ctx context.Context, userID string, items []OrderItem) (*Order, error) {
	_, span := s.tracer.Start(ctx, "database.insertOrder",
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(
			semconv.DBSystemPostgreSQL,
			semconv.DBOperation("INSERT"),
			attribute.String("db.table", "orders"),
		),
	)
	defer span.End()

	// Database insertion logic
	order := &Order{
		ID:     generateOrderID(),
		UserID: userID,
		Items:  items,
		Total:  calculateTotal(items),
	}

	span.SetAttributes(attribute.String("order.id", order.ID))
	span.SetStatus(codes.Ok, "order inserted")
	return order, nil
}

// HTTP middleware example
func TracingMiddleware() func(http.Handler) http.Handler {
	return otelhttp.NewMiddleware("http-server",
		otelhttp.WithSpanNameFormatter(func(operation string, r *http.Request) string {
			return fmt.Sprintf("%s %s", r.Method, r.URL.Path)
		}),
	)
}

// gRPC instrumentation
func GRPCTracingInterceptor() grpc.UnaryServerInterceptor {
	return otelgrpc.UnaryServerInterceptor()
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
```

### Java/Spring Boot

```java
// TracingConfiguration.java
package com.example.tracing;

import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;
import io.opentelemetry.exporter.otlp.http.trace.OtlpHttpSpanExporter;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import io.opentelemetry.sdk.trace.samplers.Sampler;
import io.opentelemetry.semconv.resource.attributes.ResourceAttributes;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.beans.factory.annotation.Value;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

@Configuration
public class TracingConfiguration {

    @Value("${spring.application.name}")
    private String serviceName;

    @Value("${otel.exporter.otlp.endpoint:http://localhost:4318/v1/traces}")
    private String otlpEndpoint;

    @Value("${otel.trace.sample.rate:0.1}")
    private double sampleRate;

    @Bean
    public OpenTelemetry openTelemetry() {
        // Create resource
        Resource resource = Resource.getDefault()
            .merge(Resource.create(Attributes.of(
                ResourceAttributes.SERVICE_NAME, serviceName,
                ResourceAttributes.SERVICE_VERSION, "1.0.0",
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT,
                    System.getenv().getOrDefault("ENVIRONMENT", "development")
            )));

        // Configure OTLP exporter
        OtlpHttpSpanExporter spanExporter = OtlpHttpSpanExporter.builder()
            .setEndpoint(otlpEndpoint)
            .setTimeout(Duration.ofSeconds(30))
            .build();

        // Configure tracer provider
        SdkTracerProvider tracerProvider = SdkTracerProvider.builder()
            .setResource(resource)
            .setSampler(Sampler.parentBased(Sampler.traceIdRatioBased(sampleRate)))
            .addSpanProcessor(BatchSpanProcessor.builder(spanExporter)
                .setMaxQueueSize(2048)
                .setMaxExportBatchSize(512)
                .setScheduleDelay(Duration.ofSeconds(5))
                .setExporterTimeout(Duration.ofSeconds(30))
                .build())
            .build();

        // Create OpenTelemetry SDK
        OpenTelemetrySdk openTelemetry = OpenTelemetrySdk.builder()
            .setTracerProvider(tracerProvider)
            .buildAndRegisterGlobal();

        // Shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            tracerProvider.close();
        }));

        return openTelemetry;
    }

    @Bean
    public Tracer tracer(OpenTelemetry openTelemetry) {
        return openTelemetry.getTracer("order-service", "1.0.0");
    }
}

// OrderService.java - Manual instrumentation
package com.example.service;

import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.List;

@Service
public class OrderService {

    @Autowired
    private Tracer tracer;

    @Autowired
    private OrderRepository orderRepository;

    public Order createOrder(String userId, List<OrderItem> items) {
        Span span = tracer.spanBuilder("createOrder")
            .setSpanKind(SpanKind.SERVER)
            .setAttribute("user.id", userId)
            .setAttribute("order.items.count", items.size())
            .startSpan();

        try (Scope scope = span.makeCurrent()) {
            // Validate inventory
            validateInventory(items);

            // Calculate total
            double total = calculateTotal(items);
            span.setAttribute("order.total", total);

            // Insert order
            Order order = insertOrder(userId, items, total);

            // Add event
            span.addEvent("order_created", Attributes.of(
                AttributeKey.stringKey("order.id"), order.getId(),
                AttributeKey.stringKey("order.status"), "pending"
            ));

            span.setStatus(StatusCode.OK);
            return order;

        } catch (Exception e) {
            span.recordException(e);
            span.setStatus(StatusCode.ERROR, e.getMessage());
            throw e;
        } finally {
            span.end();
        }
    }

    private void validateInventory(List<OrderItem> items) {
        Span span = tracer.spanBuilder("validateInventory")
            .setAttribute("inventory.items", items.size())
            .startSpan();

        try (Scope scope = span.makeCurrent()) {
            // Validation logic
            for (OrderItem item : items) {
                if (item.getQuantity() <= 0) {
                    throw new IllegalArgumentException("Invalid quantity");
                }
            }
            span.setStatus(StatusCode.OK);
        } finally {
            span.end();
        }
    }

    private Order insertOrder(String userId, List<OrderItem> items, double total) {
        Span span = tracer.spanBuilder("database.insertOrder")
            .setSpanKind(SpanKind.CLIENT)
            .setAttribute("db.system", "postgresql")
            .setAttribute("db.operation", "INSERT")
            .setAttribute("db.table", "orders")
            .startSpan();

        try (Scope scope = span.makeCurrent()) {
            Order order = new Order(userId, items, total);
            order = orderRepository.save(order);
            span.setAttribute("order.id", order.getId());
            span.setStatus(StatusCode.OK);
            return order;
        } finally {
            span.end();
        }
    }
}
```

## Trace Context Propagation

### W3C Trace Context

```text
Standard Headers:
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
            |  |                              |                  |
            |  trace-id (32 hex chars)        span-id (16 hex)  flags
            version

tracestate: vendor1=value1,vendor2=value2
```

### Implementation Examples

```typescript
// w3c-propagation.ts
import { context, propagation, trace } from '@opentelemetry/api';
import { W3CTraceContextPropagator } from '@opentelemetry/core';

export class W3CContextPropagation {
  private propagator = new W3CTraceContextPropagator();

  // Extract context from incoming HTTP headers
  extractContext(headers: Record<string, string>): Context {
    return propagation.extract(context.active(), headers);
  }

  // Inject context into outgoing HTTP headers
  injectContext(ctx: Context): Record<string, string> {
    const headers: Record<string, string> = {};
    propagation.inject(ctx, headers);
    return headers;
  }

  // HTTP client with context propagation
  async makeRequest(url: string, options: RequestInit = {}): Promise<Response> {
    const activeContext = context.active();
    const headers = this.injectContext(activeContext);

    return fetch(url, {
      ...options,
      headers: {
        ...options.headers,
        ...headers,
      },
    });
  }

  // Express middleware to extract context
  middleware() {
    return (req: any, res: any, next: any) => {
      const extractedContext = this.extractContext(req.headers);
      context.with(extractedContext, next);
    };
  }
}
```

### B3 Propagation

```python
# b3_propagation.py
from opentelemetry.propagators.b3 import B3MultiFormat, B3SingleFormat
from opentelemetry.context import Context
from typing import Dict

class B3ContextPropagation:
    """B3 propagation for Zipkin compatibility."""

    def __init__(self, multi_header: bool = True):
        self.propagator = B3MultiFormat() if multi_header else B3SingleFormat()

    def extract_context(self, headers: Dict[str, str]) -> Context:
        """Extract B3 context from headers."""
        # Multi-header format:
        # X-B3-TraceId: 4bf92f3577b34da6a3ce929d0e0e4736
        # X-B3-SpanId: 00f067aa0ba902b7
        # X-B3-ParentSpanId: <parent-id>
        # X-B3-Sampled: 1
        # X-B3-Flags: 1

        # Single-header format:
        # b3: {TraceId}-{SpanId}-{SamplingState}-{ParentSpanId}

        carrier = {k.lower(): v for k, v in headers.items()}
        return self.propagator.extract(carrier)

    def inject_context(self, context: Context) -> Dict[str, str]:
        """Inject B3 context into headers."""
        carrier = {}
        self.propagator.inject(carrier, context)
        return carrier
```

### Jaeger Propagation

```go
// jaeger_propagation.go
package propagation

import (
	"context"
	"net/http"

	"go.opentelemetry.io/contrib/propagators/jaeger"
	"go.opentelemetry.io/otel/propagation"
)

type JaegerPropagation struct {
	propagator propagation.TextMapPropagator
}

func NewJaegerPropagation() *JaegerPropagation {
	return &JaegerPropagation{
		propagator: jaeger.Jaeger{},
	}
}

// ExtractContext from Jaeger HTTP headers
// Header format: uber-trace-id: {trace-id}:{span-id}:{parent-id}:{flags}
func (j *JaegerPropagation) ExtractContext(r *http.Request) context.Context {
	return j.propagator.Extract(r.Context(), propagation.HeaderCarrier(r.Header))
}

// InjectContext into Jaeger HTTP headers
func (j *JaegerPropagation) InjectContext(ctx context.Context, r *http.Request) {
	j.propagator.Inject(ctx, propagation.HeaderCarrier(r.Header))
}

// HTTPClient with Jaeger context propagation
func (j *JaegerPropagation) HTTPClient(client *http.Client) *http.Client {
	transport := client.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}

	client.Transport = &jaegerTransport{
		base:       transport,
		propagator: j.propagator,
	}
	return client
}

type jaegerTransport struct {
	base       http.RoundTripper
	propagator propagation.TextMapPropagator
}

func (t *jaegerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	t.propagator.Inject(req.Context(), propagation.HeaderCarrier(req.Header))
	return t.base.RoundTrip(req)
}
```

## Jaeger Deployment

### Docker Compose Deployment

```yaml
# docker-compose.jaeger.yml
version: '3.8'

services:
  jaeger-all-in-one:
    image: jaegertracing/all-in-one:1.51
    container_name: jaeger
    environment:
      - COLLECTOR_OTLP_ENABLED=true
      - SPAN_STORAGE_TYPE=badger
      - BADGER_EPHEMERAL=false
      - BADGER_DIRECTORY_VALUE=/badger/data
      - BADGER_DIRECTORY_KEY=/badger/key
    ports:
      - "16686:16686"   # Jaeger UI
      - "14268:14268"   # Jaeger collector HTTP
      - "14250:14250"   # Jaeger collector gRPC
      - "4317:4317"     # OTLP gRPC
      - "4318:4318"     # OTLP HTTP
      - "6831:6831/udp" # Jaeger agent compact thrift
    volumes:
      - jaeger-badger:/badger
    networks:
      - tracing

  # Production setup with separate components
  jaeger-collector:
    image: jaegertracing/jaeger-collector:1.51
    container_name: jaeger-collector
    environment:
      - SPAN_STORAGE_TYPE=elasticsearch
      - ES_SERVER_URLS=http://elasticsearch:9200
      - ES_TAGS_AS_FIELDS_ALL=true
      - COLLECTOR_OTLP_ENABLED=true
      - COLLECTOR_QUEUE_SIZE=10000
      - COLLECTOR_NUM_WORKERS=100
    ports:
      - "14269:14269"   # Admin port
      - "14268:14268"   # HTTP
      - "14250:14250"   # gRPC
      - "4317:4317"     # OTLP gRPC
      - "4318:4318"     # OTLP HTTP
    depends_on:
      - elasticsearch
    networks:
      - tracing

  jaeger-query:
    image: jaegertracing/jaeger-query:1.51
    container_name: jaeger-query
    environment:
      - SPAN_STORAGE_TYPE=elasticsearch
      - ES_SERVER_URLS=http://elasticsearch:9200
      - QUERY_BASE_PATH=/jaeger
    ports:
      - "16686:16686"   # UI
      - "16687:16687"   # Admin
    depends_on:
      - elasticsearch
    networks:
      - tracing

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
    volumes:
      - es-data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - tracing

volumes:
  jaeger-badger:
  es-data:

networks:
  tracing:
    driver: bridge
```

### Kubernetes Deployment

```yaml
# jaeger-k8s.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: observability

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
  namespace: observability
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:1.51
        env:
        - name: COLLECTOR_OTLP_ENABLED
          value: "true"
        - name: SPAN_STORAGE_TYPE
          value: "badger"
        - name: BADGER_EPHEMERAL
          value: "false"
        - name: BADGER_DIRECTORY_VALUE
          value: "/badger/data"
        - name: BADGER_DIRECTORY_KEY
          value: "/badger/key"
        ports:
        - containerPort: 16686
          name: ui
        - containerPort: 4317
          name: otlp-grpc
        - containerPort: 4318
          name: otlp-http
        - containerPort: 14250
          name: grpc
        - containerPort: 14268
          name: http
        volumeMounts:
        - name: badger-data
          mountPath: /badger
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: badger-data
        persistentVolumeClaim:
          claimName: jaeger-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: jaeger
  namespace: observability
spec:
  type: LoadBalancer
  selector:
    app: jaeger
  ports:
  - name: ui
    port: 16686
    targetPort: 16686
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
  - name: otlp-http
    port: 4318
    targetPort: 4318
  - name: grpc
    port: 14250
    targetPort: 14250

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: jaeger-pvc
  namespace: observability
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
```

### Jaeger Operator Deployment

```yaml
# jaeger-operator.yaml
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: production-jaeger
  namespace: observability
spec:
  strategy: production
  storage:
    type: elasticsearch
    options:
      es:
        server-urls: http://elasticsearch:9200
        index-prefix: jaeger
        num-shards: 5
        num-replicas: 1
    esIndexCleaner:
      enabled: true
      numberOfDays: 7
      schedule: "55 23 * * *"
  collector:
    replicas: 3
    maxReplicas: 10
    resources:
      requests:
        cpu: 500m
        memory: 1Gi
      limits:
        cpu: 2000m
        memory: 4Gi
    options:
      collector:
        queue-size: 10000
        num-workers: 100
  query:
    replicas: 2
    resources:
      requests:
        cpu: 250m
        memory: 512Mi
      limits:
        cpu: 1000m
        memory: 2Gi
  ingress:
    enabled: true
    annotations:
      kubernetes.io/ingress.class: nginx
      cert-manager.io/cluster-issuer: letsencrypt-prod
    hosts:
    - jaeger.example.com
    tls:
    - secretName: jaeger-tls
      hosts:
      - jaeger.example.com
```

## Tempo Integration

### Tempo Configuration

```yaml
# tempo-config.yaml
auth_enabled: false

server:
  http_listen_port: 3200
  grpc_listen_port: 9096

distributor:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317
        http:
          endpoint: 0.0.0.0:4318
    jaeger:
      protocols:
        thrift_http:
          endpoint: 0.0.0.0:14268
        grpc:
          endpoint: 0.0.0.0:14250

ingester:
  max_block_duration: 5m
  trace_idle_period: 10s
  max_block_bytes: 1_000_000

compactor:
  compaction:
    block_retention: 168h  # 7 days

storage:
  trace:
    backend: s3
    s3:
      bucket: tempo-traces
      endpoint: s3.amazonaws.com
      region: us-east-1
    wal:
      path: /var/tempo/wal
    block:
      bloom_filter_false_positive: 0.05
      index_downsample_bytes: 1000
      encoding: zstd

metrics_generator:
  registry:
    external_labels:
      source: tempo
      cluster: production
  storage:
    path: /var/tempo/generator/wal
    remote_write:
      - url: http://prometheus:9090/api/v1/write
        send_exemplars: true

overrides:
  metrics_generator_processors:
    - service-graphs
    - span-metrics
```

### Docker Compose with Grafana

```yaml
# docker-compose.tempo.yml
version: '3.8'

services:
  tempo:
    image: grafana/tempo:2.3.0
    container_name: tempo
    command: [ "-config.file=/etc/tempo.yaml" ]
    volumes:
      - ./tempo-config.yaml:/etc/tempo.yaml
      - tempo-data:/var/tempo
    ports:
      - "3200:3200"     # Tempo
      - "4317:4317"     # OTLP gRPC
      - "4318:4318"     # OTLP HTTP
      - "14250:14250"   # Jaeger gRPC
    networks:
      - observability

  grafana:
    image: grafana/grafana:10.2.0
    container_name: grafana
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
      - GF_FEATURE_TOGGLES_ENABLE=traceqlEditor
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana-datasources.yaml:/etc/grafana/provisioning/datasources/datasources.yaml
    ports:
      - "3000:3000"
    depends_on:
      - tempo
      - prometheus
    networks:
      - observability

  prometheus:
    image: prom/prometheus:v2.48.0
    container_name: prometheus
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --enable-feature=exemplar-storage
      - --web.enable-remote-write-receiver
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - observability

volumes:
  tempo-data:
  grafana-data:
  prometheus-data:

networks:
  observability:
    driver: bridge
```

### Grafana Datasource Configuration

```yaml
# grafana-datasources.yaml
apiVersion: 1

datasources:
  - name: Tempo
    type: tempo
    access: proxy
    url: http://tempo:3200
    uid: tempo
    jsonData:
      httpMethod: GET
      tracesToLogs:
        datasourceUid: 'loki'
        tags: ['job', 'instance', 'pod', 'namespace']
        mappedTags: [{ key: 'service.name', value: 'service' }]
        mapTagNamesEnabled: true
        spanStartTimeShift: '1h'
        spanEndTimeShift: '-1h'
        filterByTraceID: true
        filterBySpanID: false
      tracesToMetrics:
        datasourceUid: 'prometheus'
        tags: [{ key: 'service.name', value: 'service' }]
        queries:
          - name: 'Sample query'
            query: 'sum(rate(traces_spanmetrics_latency_bucket{$__tags}[5m]))'
      serviceMap:
        datasourceUid: 'prometheus'
      nodeGraph:
        enabled: true
      search:
        hide: false
      lokiSearch:
        datasourceUid: 'loki'

  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    uid: prometheus
    jsonData:
      httpMethod: POST
      exemplarTraceIdDestinations:
        - name: trace_id
          datasourceUid: tempo

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    uid: loki
    jsonData:
      derivedFields:
        - datasourceUid: tempo
          matcherRegex: "trace_id=(\\w+)"
          name: TraceID
          url: '$${__value.raw}'
```

### TraceQL Queries

```traceql
# Find slow traces
{ duration > 1s }

# Find traces with errors
{ status = error }

# Find traces for specific service
{ resource.service.name = "order-service" }

# Complex query
{
  resource.service.name = "order-service" &&
  span.http.status_code >= 500 &&
  duration > 500ms
} | select(span.http.route, span.http.status_code)

# Aggregate metrics
{ resource.service.name = "api-gateway" }
| by(span.http.route)
| rate()
```

## Trace Sampling Strategies

### Head-Based Sampling

```typescript
// head-based-sampling.ts
import { Sampler, SamplingDecision, SamplingResult } from '@opentelemetry/sdk-trace-base';
import { Attributes, Context, SpanKind } from '@opentelemetry/api';

export class CustomHeadSampler implements Sampler {
  private defaultRate: number;
  private rules: SamplingRule[];

  constructor(defaultRate: number, rules: SamplingRule[] = []) {
    this.defaultRate = defaultRate;
    this.rules = rules;
  }

  shouldSample(
    context: Context,
    traceId: string,
    spanName: string,
    spanKind: SpanKind,
    attributes: Attributes
  ): SamplingResult {
    // Check custom rules first
    for (const rule of this.rules) {
      if (rule.matches(spanName, attributes)) {
        return {
          decision: rule.sample ? SamplingDecision.RECORD_AND_SAMPLED : SamplingDecision.NOT_RECORD,
          attributes: rule.attributes,
        };
      }
    }

    // Apply default rate-based sampling
    const hash = this.hashTraceId(traceId);
    const decision = hash < this.defaultRate
      ? SamplingDecision.RECORD_AND_SAMPLED
      : SamplingDecision.NOT_RECORD;

    return { decision };
  }

  toString(): string {
    return `CustomHeadSampler{defaultRate=${this.defaultRate}}`;
  }

  private hashTraceId(traceId: string): number {
    // Simple hash function for trace ID
    const hash = traceId.split('').reduce((acc, char) => {
      return ((acc << 5) - acc) + char.charCodeAt(0);
    }, 0);
    return Math.abs(hash) / Number.MAX_SAFE_INTEGER;
  }
}

export interface SamplingRule {
  matches(spanName: string, attributes: Attributes): boolean;
  sample: boolean;
  attributes?: Attributes;
}

// Example rules
export const samplingRules: SamplingRule[] = [
  {
    // Always sample errors
    matches: (spanName, attrs) => attrs['http.status_code'] >= 400,
    sample: true,
    attributes: { 'sampling.rule': 'error-always' },
  },
  {
    // Always sample slow requests
    matches: (spanName, attrs) => attrs['duration.ms'] > 1000,
    sample: true,
    attributes: { 'sampling.rule': 'slow-always' },
  },
  {
    // Never sample health checks
    matches: (spanName, attrs) => spanName.includes('health'),
    sample: false,
  },
  {
    // High rate for critical endpoints
    matches: (spanName, attrs) => attrs['http.route'] === '/api/checkout',
    sample: true,
    attributes: { 'sampling.rule': 'critical-endpoint' },
  },
];
```

### Tail-Based Sampling

```python
# tail_based_sampling.py
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from typing import List, Dict, Callable
from collections import defaultdict
import time
import threading

class TailBasedSamplingProcessor(SpanProcessor):
    """Tail-based sampling decides whether to keep traces after completion."""

    def __init__(
        self,
        exporter: SpanExporter,
        decision_wait_ms: int = 30000,
        num_traces: int = 10000,
        policies: List['SamplingPolicy'] = None
    ):
        self.exporter = exporter
        self.decision_wait_ms = decision_wait_ms
        self.num_traces = num_traces
        self.policies = policies or []

        # Buffer for incomplete traces
        self.trace_buffer: Dict[str, List[ReadableSpan]] = defaultdict(list)
        self.trace_timestamps: Dict[str, float] = {}

        # Start background thread for decision making
        self.lock = threading.Lock()
        self.running = True
        self.decision_thread = threading.Thread(target=self._decision_loop, daemon=True)
        self.decision_thread.start()

    def on_start(self, span: ReadableSpan, parent_context) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        trace_id = format(span.context.trace_id, '032x')

        with self.lock:
            self.trace_buffer[trace_id].append(span)
            if trace_id not in self.trace_timestamps:
                self.trace_timestamps[trace_id] = time.time()

    def _decision_loop(self):
        """Background thread to make sampling decisions."""
        while self.running:
            time.sleep(1)
            current_time = time.time()

            with self.lock:
                traces_to_decide = []

                for trace_id, timestamp in list(self.trace_timestamps.items()):
                    age_ms = (current_time - timestamp) * 1000

                    if age_ms >= self.decision_wait_ms:
                        traces_to_decide.append(trace_id)

                # Make sampling decisions
                for trace_id in traces_to_decide:
                    spans = self.trace_buffer.pop(trace_id, [])
                    del self.trace_timestamps[trace_id]

                    if self._should_sample_trace(spans):
                        self.exporter.export(spans)

    def _should_sample_trace(self, spans: List[ReadableSpan]) -> bool:
        """Apply policies to decide if trace should be sampled."""
        for policy in self.policies:
            if policy.evaluate(spans):
                return True
        return False

    def shutdown(self):
        self.running = False
        self.decision_thread.join()
        self.exporter.shutdown()

class SamplingPolicy:
    """Base class for tail-based sampling policies."""

    def evaluate(self, spans: List[ReadableSpan]) -> bool:
        raise NotImplementedError

class ErrorPolicy(SamplingPolicy):
    """Sample all traces with errors."""

    def evaluate(self, spans: List[ReadableSpan]) -> bool:
        return any(span.status.status_code == StatusCode.ERROR for span in spans)

class LatencyPolicy(SamplingPolicy):
    """Sample traces exceeding latency threshold."""

    def __init__(self, threshold_ms: float):
        self.threshold_ms = threshold_ms

    def evaluate(self, spans: List[ReadableSpan]) -> bool:
        if not spans:
            return False

        # Find root span
        root_span = min(spans, key=lambda s: s.start_time)
        end_span = max(spans, key=lambda s: s.end_time)

        duration_ms = (end_span.end_time - root_span.start_time) / 1_000_000
        return duration_ms > self.threshold_ms

class AttributePolicy(SamplingPolicy):
    """Sample traces matching specific attributes."""

    def __init__(self, attribute_matcher: Callable[[Dict], bool]):
        self.attribute_matcher = attribute_matcher

    def evaluate(self, spans: List[ReadableSpan]) -> bool:
        for span in spans:
            if self.attribute_matcher(span.attributes):
                return True
        return False

class RateLimitingPolicy(SamplingPolicy):
    """Limit sampling rate to N traces per second."""

    def __init__(self, traces_per_second: int):
        self.traces_per_second = traces_per_second
        self.window_start = time.time()
        self.count = 0

    def evaluate(self, spans: List[ReadableSpan]) -> bool:
        current_time = time.time()

        # Reset window
        if current_time - self.window_start >= 1.0:
            self.window_start = current_time
            self.count = 0

        if self.count < self.traces_per_second:
            self.count += 1
            return True

        return False
```

### Probability Sampling

```go
// probability_sampling.go
package sampling

import (
	"crypto/rand"
	"encoding/binary"
	"math"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// ProbabilitySampler implements probability-based sampling
type ProbabilitySampler struct {
	probability float64
	threshold   uint64
}

// NewProbabilitySampler creates a new probability sampler
func NewProbabilitySampler(probability float64) *ProbabilitySampler {
	if probability < 0.0 || probability > 1.0 {
		probability = 0.1 // Default 10%
	}

	return &ProbabilitySampler{
		probability: probability,
		threshold:   uint64(probability * math.MaxUint64),
	}
}

func (s *ProbabilitySampler) ShouldSample(parameters sdktrace.SamplingParameters) sdktrace.SamplingResult {
	// Extract trace ID
	traceID := parameters.TraceID

	// Convert trace ID to uint64 for comparison
	var hashValue uint64
	for i := 0; i < 8; i++ {
		hashValue <<= 8
		hashValue |= uint64(traceID[i])
	}

	decision := sdktrace.Drop
	if hashValue <= s.threshold {
		decision = sdktrace.RecordAndSample
	}

	return sdktrace.SamplingResult{
		Decision:   decision,
		Attributes: []attribute.KeyValue{},
	}
}

func (s *ProbabilitySampler) Description() string {
	return fmt.Sprintf("ProbabilitySampler{%.2f}", s.probability)
}

// AdaptiveSampler adjusts sampling rate based on traffic
type AdaptiveSampler struct {
	minProbability float64
	maxProbability float64
	targetRate     int
	currentRate    int
	windowStart    time.Time
	mu             sync.Mutex
}

func NewAdaptiveSampler(targetTracesPerSecond int) *AdaptiveSampler {
	return &AdaptiveSampler{
		minProbability: 0.001,  // 0.1%
		maxProbability: 1.0,    // 100%
		targetRate:     targetTracesPerSecond,
		windowStart:    time.Now(),
	}
}

func (s *AdaptiveSampler) ShouldSample(parameters sdktrace.SamplingParameters) sdktrace.SamplingResult {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Adjust probability every second
	if time.Since(s.windowStart) >= time.Second {
		s.adjustProbability()
		s.windowStart = time.Now()
		s.currentRate = 0
	}

	// Apply current probability
	probability := s.getCurrentProbability()
	sampler := NewProbabilitySampler(probability)
	result := sampler.ShouldSample(parameters)

	if result.Decision == sdktrace.RecordAndSample {
		s.currentRate++
	}

	return result
}

func (s *AdaptiveSampler) adjustProbability() {
	if s.currentRate > s.targetRate {
		// Reduce sampling
		newProb := s.getCurrentProbability() * 0.9
		if newProb < s.minProbability {
			newProb = s.minProbability
		}
		s.setCurrentProbability(newProb)
	} else if s.currentRate < s.targetRate {
		// Increase sampling
		newProb := s.getCurrentProbability() * 1.1
		if newProb > s.maxProbability {
			newProb = s.maxProbability
		}
		s.setCurrentProbability(newProb)
	}
}
```

## Span Attributes and Semantics

### Semantic Conventions

```typescript
// semantic-conventions.ts
import { SemanticAttributes } from '@opentelemetry/semantic-conventions';
import { Span } from '@opentelemetry/api';

export class SpanAttributesBuilder {
  // HTTP attributes
  static setHttpAttributes(span: Span, req: Request, res: Response): void {
    span.setAttributes({
      [SemanticAttributes.HTTP_METHOD]: req.method,
      [SemanticAttributes.HTTP_URL]: req.url,
      [SemanticAttributes.HTTP_TARGET]: req.path,
      [SemanticAttributes.HTTP_HOST]: req.hostname,
      [SemanticAttributes.HTTP_SCHEME]: req.protocol,
      [SemanticAttributes.HTTP_STATUS_CODE]: res.statusCode,
      [SemanticAttributes.HTTP_USER_AGENT]: req.get('user-agent') || '',
      [SemanticAttributes.HTTP_REQUEST_CONTENT_LENGTH]: req.get('content-length') || 0,
      [SemanticAttributes.HTTP_RESPONSE_CONTENT_LENGTH]: res.get('content-length') || 0,
      [SemanticAttributes.HTTP_ROUTE]: req.route?.path || '',
      [SemanticAttributes.HTTP_CLIENT_IP]: req.ip || '',
    });
  }

  // Database attributes
  static setDatabaseAttributes(span: Span, operation: string, table: string, statement?: string): void {
    span.setAttributes({
      [SemanticAttributes.DB_SYSTEM]: 'postgresql',
      [SemanticAttributes.DB_CONNECTION_STRING]: 'postgresql://localhost:5432/mydb',
      [SemanticAttributes.DB_USER]: 'app_user',
      [SemanticAttributes.DB_NAME]: 'production',
      [SemanticAttributes.DB_STATEMENT]: statement || '',
      [SemanticAttributes.DB_OPERATION]: operation,
      [SemanticAttributes.DB_SQL_TABLE]: table,
    });
  }

  // RPC/gRPC attributes
  static setRpcAttributes(span: Span, service: string, method: string): void {
    span.setAttributes({
      [SemanticAttributes.RPC_SYSTEM]: 'grpc',
      [SemanticAttributes.RPC_SERVICE]: service,
      [SemanticAttributes.RPC_METHOD]: method,
      [SemanticAttributes.RPC_GRPC_STATUS_CODE]: 0,
    });
  }

  // Messaging attributes
  static setMessagingAttributes(span: Span, system: string, destination: string, operation: string): void {
    span.setAttributes({
      [SemanticAttributes.MESSAGING_SYSTEM]: system,
      [SemanticAttributes.MESSAGING_DESTINATION]: destination,
      [SemanticAttributes.MESSAGING_DESTINATION_KIND]: 'queue',
      [SemanticAttributes.MESSAGING_OPERATION]: operation,
      [SemanticAttributes.MESSAGING_MESSAGE_ID]: 'msg-123',
    });
  }

  // Custom business attributes
  static setBusinessAttributes(span: Span, userId: string, tenantId: string, feature: string): void {
    span.setAttributes({
      'user.id': userId,
      'user.tenant_id': tenantId,
      'business.feature': feature,
      'business.transaction_type': 'purchase',
      'business.amount': 99.99,
      'business.currency': 'USD',
    });
  }
}
```

## Trace Correlation

### Logs-Traces-Metrics Correlation

```typescript
// correlation.ts
import { trace, context } from '@opentelemetry/api';
import { Logger } from 'winston';

export class CorrelationService {
  constructor(private logger: Logger) {}

  // Add trace context to logs
  logWithTrace(level: string, message: string, metadata: any = {}): void {
    const activeSpan = trace.getSpan(context.active());

    if (activeSpan) {
      const spanContext = activeSpan.spanContext();
      this.logger.log(level, message, {
        ...metadata,
        trace_id: spanContext.traceId,
        span_id: spanContext.spanId,
        trace_flags: spanContext.traceFlags,
      });
    } else {
      this.logger.log(level, message, metadata);
    }
  }

  // Extract trace context for metrics
  getTraceLabels(): Record<string, string> {
    const activeSpan = trace.getSpan(context.active());

    if (activeSpan) {
      const spanContext = activeSpan.spanContext();
      return {
        trace_id: spanContext.traceId,
        span_id: spanContext.spanId,
      };
    }

    return {};
  }

  // Unified observability context
  getObservabilityContext(): ObservabilityContext {
    const activeSpan = trace.getSpan(context.active());

    if (activeSpan) {
      const spanContext = activeSpan.spanContext();
      const attrs = activeSpan.attributes || {};

      return {
        trace_id: spanContext.traceId,
        span_id: spanContext.spanId,
        service_name: attrs['service.name'] as string,
        operation_name: activeSpan.name,
        user_id: attrs['user.id'] as string,
        tenant_id: attrs['tenant.id'] as string,
      };
    }

    return {};
  }
}

export interface ObservabilityContext {
  trace_id?: string;
  span_id?: string;
  service_name?: string;
  operation_name?: string;
  user_id?: string;
  tenant_id?: string;
}
```

### Structured Logging with Trace Context

```python
# structured_logging.py
import logging
import json
from opentelemetry import trace
from opentelemetry.trace import get_current_span

class TraceContextFormatter(logging.Formatter):
    """Add trace context to log records."""

    def format(self, record):
        span = get_current_span()

        if span and span.get_span_context().is_valid:
            span_context = span.get_span_context()
            record.trace_id = format(span_context.trace_id, '032x')
            record.span_id = format(span_context.span_id, '016x')
            record.trace_flags = span_context.trace_flags
        else:
            record.trace_id = '0' * 32
            record.span_id = '0' * 16
            record.trace_flags = 0

        return super().format(record)

class JSONFormatter(TraceContextFormatter):
    """Format logs as JSON with trace correlation."""

    def format(self, record):
        super().format(record)

        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'trace_id': record.trace_id,
            'span_id': record.span_id,
            'trace_flags': record.trace_flags,
        }

        # Add custom fields
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id

        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id

        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)

# Configure logger
def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger
```

## Service Mesh Tracing

### Istio Configuration

```yaml
# istio-tracing.yaml
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: istio-tracing
spec:
  meshConfig:
    enableTracing: true
    defaultConfig:
      tracing:
        sampling: 10.0  # 10% sampling
        custom_tags:
          environment:
            literal:
              value: production
          region:
            environment:
              name: REGION
        zipkin:
          address: tempo.observability.svc.cluster.local:9411
    extensionProviders:
      - name: tempo
        opentelemetry:
          service: tempo.observability.svc.cluster.local
          port: 4317
      - name: jaeger
        opentelemetry:
          service: jaeger-collector.observability.svc.cluster.local
          port: 4317

  values:
    global:
      proxy:
        tracer: "opentelemetry"
    pilot:
      traceSampling: 10.0

---
apiVersion: telemetry.istio.io/v1alpha1
kind: Telemetry
metadata:
  name: mesh-tracing
  namespace: istio-system
spec:
  tracing:
    - providers:
        - name: tempo
      randomSamplingPercentage: 10.0
      customTags:
        user_id:
          header:
            name: x-user-id
        tenant_id:
          header:
            name: x-tenant-id
        request_id:
          header:
            name: x-request-id
```

### Linkerd Configuration

```yaml
# linkerd-tracing.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: linkerd-config
  namespace: linkerd
data:
  config: |
    {
      "tracing": {
        "enabled": true,
        "collector": "tempo-collector.observability.svc.cluster.local:4317",
        "sampling_rate": 0.1
      }
    }

---
apiVersion: policy.linkerd.io/v1alpha1
kind: HTTPRoute
metadata:
  name: traced-route
  namespace: default
spec:
  parentRefs:
    - name: api-gateway
      kind: Service
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /api
      filters:
        - type: RequestHeaderModifier
          requestHeaderModifier:
            add:
              - name: l5d-trace-id
                value: "%TRACE_ID%"
```

## Performance Optimization

### Batching and Buffering

```python
# performance_optimization.py
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from typing import Sequence
import asyncio
import time

class OptimizedBatchProcessor(BatchSpanProcessor):
    """Optimized batch processor with dynamic batching."""

    def __init__(
        self,
        exporter: SpanExporter,
        max_queue_size: int = 4096,
        max_export_batch_size: int = 512,
        schedule_delay_millis: int = 5000,
        export_timeout_millis: int = 30000,
        adaptive_batching: bool = True
    ):
        super().__init__(
            exporter,
            max_queue_size=max_queue_size,
            max_export_batch_size=max_export_batch_size,
            schedule_delay_millis=schedule_delay_millis,
            export_timeout_millis=export_timeout_millis
        )
        self.adaptive_batching = adaptive_batching
        self.export_latencies = []

    def _export_batch(self, spans: Sequence[ReadableSpan]) -> None:
        """Export batch with performance monitoring."""
        start_time = time.time()

        try:
            super()._export_batch(spans)

            # Track export performance
            latency_ms = (time.time() - start_time) * 1000
            self.export_latencies.append(latency_ms)

            # Adjust batch size dynamically
            if self.adaptive_batching:
                self._adjust_batch_size(latency_ms)

        except Exception as e:
            print(f"Export failed: {e}")

    def _adjust_batch_size(self, latency_ms: float):
        """Dynamically adjust batch size based on latency."""
        if latency_ms > 1000:  # Too slow
            self.max_export_batch_size = max(128, self.max_export_batch_size // 2)
        elif latency_ms < 100:  # Can handle more
            self.max_export_batch_size = min(1024, self.max_export_batch_size * 2)

class CompressionExporter(SpanExporter):
    """Exporter with compression for reduced network overhead."""

    def __init__(self, base_exporter: SpanExporter, compression: str = 'gzip'):
        self.base_exporter = base_exporter
        self.compression = compression

    def export(self, spans: Sequence[ReadableSpan]):
        # Compress spans before export
        compressed_spans = self._compress_spans(spans)
        return self.base_exporter.export(compressed_spans)

    def _compress_spans(self, spans):
        # Implementation of span compression
        return spans

    def shutdown(self):
        self.base_exporter.shutdown()
```

### Resource Management

```go
// resource_optimization.go
package optimization

import (
	"context"
	"runtime"
	"sync"
	"time"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// ResourceOptimizedProvider manages tracing resources efficiently
type ResourceOptimizedProvider struct {
	provider *sdktrace.TracerProvider
	metrics  *PerformanceMetrics
	mu       sync.RWMutex
}

type PerformanceMetrics struct {
	SpansCreated    int64
	SpansExported   int64
	SpansDropped    int64
	ExportLatencyMs []float64
	MemoryUsageMB   float64
}

func NewResourceOptimizedProvider(provider *sdktrace.TracerProvider) *ResourceOptimizedProvider {
	rop := &ResourceOptimizedProvider{
		provider: provider,
		metrics:  &PerformanceMetrics{},
	}

	// Start resource monitoring
	go rop.monitorResources()

	return rop
}

func (rop *ResourceOptimizedProvider) monitorResources() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)

		rop.mu.Lock()
		rop.metrics.MemoryUsageMB = float64(m.Alloc) / 1024 / 1024

		// Trigger GC if memory usage is high
		if rop.metrics.MemoryUsageMB > 500 {
			runtime.GC()
		}
		rop.mu.Unlock()
	}
}

func (rop *ResourceOptimizedProvider) GetMetrics() PerformanceMetrics {
	rop.mu.RLock()
	defer rop.mu.RUnlock()
	return *rop.metrics
}

// ConnectionPooling for exporters
type PooledExporter struct {
	exporters []sdktrace.SpanExporter
	current   int
	mu        sync.Mutex
}

func NewPooledExporter(size int, factory func() sdktrace.SpanExporter) *PooledExporter {
	exporters := make([]sdktrace.SpanExporter, size)
	for i := 0; i < size; i++ {
		exporters[i] = factory()
	}

	return &PooledExporter{
		exporters: exporters,
	}
}

func (pe *PooledExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	pe.mu.Lock()
	exporter := pe.exporters[pe.current]
	pe.current = (pe.current + 1) % len(pe.exporters)
	pe.mu.Unlock()

	return exporter.ExportSpans(ctx, spans)
}

func (pe *PooledExporter) Shutdown(ctx context.Context) error {
	for _, exporter := range pe.exporters {
		if err := exporter.Shutdown(ctx); err != nil {
			return err
		}
	}
	return nil
}
```

This comprehensive guide provides production-ready distributed tracing implementations across multiple languages and platforms, complete with deployment configurations, sampling strategies, and performance optimizations for building observable microservices systems.
