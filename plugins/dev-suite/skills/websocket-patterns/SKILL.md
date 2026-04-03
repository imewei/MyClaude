---
name: websocket-patterns
description: Build real-time communication with WebSocket, Server-Sent Events, and gRPC streaming including connection management, reconnection strategies, room-based messaging, and scaling with Redis pub/sub. Use when implementing chat, live updates, notifications, or bi-directional streaming.
---

# WebSocket Patterns

## Expert Agent

For real-time architecture, streaming protocols, and scalable messaging design, delegate to:

- **`software-architect`**: Designs real-time communication architectures with protocol selection and scaling strategies.
  - *Location*: `plugins/dev-suite/agents/software-architect.md`


## Protocol Comparison

| Feature | WebSocket | SSE | gRPC Streaming |
|---------|-----------|-----|----------------|
| Direction | Bidirectional | Server-to-client | Bidirectional |
| Protocol | WS (TCP) | HTTP/1.1 | HTTP/2 |
| Reconnection | Manual | Built-in | Manual |
| Binary support | Yes | No (text only) | Yes (protobuf) |
| Browser support | All modern | All modern | Via grpc-web |
| Use case | Chat, gaming | Live feeds, notifications | Microservice streaming |


## WebSocket Server (Node.js)

```typescript
import { WebSocketServer, WebSocket } from "ws";
import { createClient } from "redis";

const wss = new WebSocketServer({ port: 8080 });
const rooms = new Map<string, Set<WebSocket>>();

wss.on("connection", (ws: WebSocket, req) => {
  const userId = authenticateConnection(req);
  if (!userId) {
    ws.close(4001, "Unauthorized");
    return;
  }

  ws.on("message", (data: Buffer) => {
    const message = JSON.parse(data.toString());
    handleMessage(ws, userId, message);
  });

  ws.on("close", () => {
    removeFromAllRooms(ws);
  });

  ws.on("pong", () => {
    (ws as any).isAlive = true;
  });

  (ws as any).isAlive = true;
});

// Heartbeat to detect dead connections
const heartbeat = setInterval(() => {
  wss.clients.forEach((ws) => {
    if (!(ws as any).isAlive) {
      ws.terminate();
      return;
    }
    (ws as any).isAlive = false;
    ws.ping();
  });
}, 30000);
```


## Room-Based Messaging

```typescript
function joinRoom(ws: WebSocket, roomId: string): void {
  if (!rooms.has(roomId)) {
    rooms.set(roomId, new Set());
  }
  rooms.get(roomId)!.add(ws);
}

function broadcastToRoom(roomId: string, message: object, exclude?: WebSocket): void {
  const members = rooms.get(roomId);
  if (!members) return;
  const payload = JSON.stringify(message);
  for (const client of members) {
    if (client !== exclude && client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  }
}
```


## Client Reconnection Strategy

```typescript
class ReconnectingWebSocket {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxAttempts = 10;
  private baseDelay = 1000;

  constructor(private url: string) {
    this.connect();
  }

  private connect(): void {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.resubscribe();
    };

    this.ws.onclose = (event) => {
      if (event.code !== 1000 && this.reconnectAttempts < this.maxAttempts) {
        const delay = this.baseDelay * Math.pow(2, this.reconnectAttempts);
        const jitter = delay * 0.1 * Math.random();
        setTimeout(() => this.connect(), delay + jitter);
        this.reconnectAttempts++;
      }
    };

    this.ws.onmessage = (event) => {
      this.handleMessage(JSON.parse(event.data));
    };
  }

}
```

## Scaling with Redis Pub/Sub

```typescript
import { createClient } from "redis";

const publisher = createClient();
const subscriber = createClient();

async function setupRedisScaling(): Promise<void> {
  await publisher.connect();
  await subscriber.connect();

  await subscriber.subscribe("room:*", (message, channel) => {
    const roomId = channel.split(":")[1];
    const parsed = JSON.parse(message);
    broadcastToRoom(roomId, parsed);
  });
}

function publishToRoom(roomId: string, message: object): void {
  publisher.publish(`room:${roomId}`, JSON.stringify(message));
}
```

## Message Protocol Design

```typescript
interface WsMessage {
  type: string;
  payload: Record<string, unknown>;
  id?: string;       // For request/response correlation
  timestamp?: number;
}

// Examples
{ "type": "join_room", "payload": { "roomId": "chat-123" } }
{ "type": "message", "payload": { "roomId": "chat-123", "text": "hello" }, "id": "msg-1" }
{ "type": "ack", "payload": { "messageId": "msg-1" } }
{ "type": "error", "payload": { "code": 4001, "message": "Not authorized" } }
```


## Design Checklist

- [ ] Protocol chosen based on communication pattern (WS/SSE/gRPC)
- [ ] Authentication enforced on connection establishment
- [ ] Heartbeat/ping-pong configured for dead connection detection
- [ ] Client reconnection with exponential backoff and jitter
- [ ] Room-based or topic-based message routing implemented
- [ ] Redis pub/sub for multi-server scaling
- [ ] Message protocol documented with type definitions
- [ ] Connection limits and rate limiting enforced
