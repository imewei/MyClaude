# Log Aggregation Architecture and Implementation Guide

## Table of Contents

1. [Log Aggregation Architecture](#log-aggregation-architecture)
2. [Fluentd/Fluent Bit Configuration](#fluentd-fluent-bit-configuration)
3. [Elasticsearch Configuration](#elasticsearch-configuration)
4. [Kibana Setup](#kibana-setup)
5. [Structured Logging Libraries](#structured-logging-libraries)
6. [Log Correlation](#log-correlation)
7. [Retention and Archival](#retention-and-archival)
8. [Log-Based Metrics](#log-based-metrics)
9. [ELK vs Loki Comparison](#elk-vs-loki-comparison)
10. [Performance Optimization](#performance-optimization)

---

## Log Aggregation Architecture

### Architecture Overview

```yaml
# Modern Log Aggregation Architecture
architecture:
  components:
    - name: "Log Producers"
      types:
        - application_logs
        - system_logs
        - container_logs
        - access_logs
        - audit_logs

    - name: "Collection Layer"
      options:
        - fluent_bit  # Lightweight, edge collection
        - fluentd     # Heavy processing, aggregation
        - vector      # High-performance alternative

    - name: "Processing Layer"
      functions:
        - parsing
        - filtering
        - enrichment
        - transformation
        - routing

    - name: "Storage Layer"
      backends:
        - elasticsearch  # Full-text search
        - loki          # Label-based
        - s3            # Long-term archival
        - clickhouse    # Analytics

    - name: "Query Layer"
      interfaces:
        - kibana
        - grafana
        - api_endpoints
```

### Multi-Tier Pipeline Design

```yaml
# Three-Tier Architecture
tiers:
  edge_tier:
    component: fluent-bit
    location: application_hosts
    responsibilities:
      - lightweight_collection
      - basic_filtering
      - buffering
      - forward_to_aggregators
    resources:
      cpu: "100m"
      memory: "128Mi"

  aggregation_tier:
    component: fluentd
    location: dedicated_nodes
    responsibilities:
      - receive_from_edge
      - heavy_parsing
      - enrichment
      - routing
      - buffering
    resources:
      cpu: "1000m"
      memory: "2Gi"

  storage_tier:
    component: elasticsearch
    location: data_cluster
    responsibilities:
      - indexing
      - search
      - analytics
      - retention
    resources:
      cpu: "4000m"
      memory: "8Gi"
```

### High-Availability Architecture

```yaml
# HA Log Pipeline Configuration
ha_design:
  edge_collection:
    deployment: daemonset
    redundancy: per_node
    buffer:
      type: disk
      size: 5GB
      flush_interval: 30s

  aggregation_cluster:
    replicas: 3
    load_balancing: round_robin
    buffer:
      type: disk
      size: 20GB
      flush_interval: 60s

  storage_cluster:
    master_nodes: 3
    data_nodes: 6
    replication_factor: 2
    shard_allocation:
      total_shards_per_node: 1000
```

---

## Fluentd/Fluent Bit Configuration

### Fluent Bit Configuration

```ini
# fluent-bit.conf - Edge Collection Agent
[SERVICE]
    Flush                     5
    Daemon                    Off
    Log_Level                 info
    Parsers_File              parsers.conf
    Plugins_File              plugins.conf
    HTTP_Server               On
    HTTP_Listen               0.0.0.0
    HTTP_Port                 2020
    Storage.metrics           On
    Storage.path              /var/log/flb-storage/
    Storage.sync              normal
    Storage.checksum          Off
    Storage.max_chunks_up     128
    Storage.backlog.mem_limit 5M

# Tail Application Logs
[INPUT]
    Name                      tail
    Path                      /var/log/app/*.log
    Parser                    json
    Tag                       app.*
    Refresh_Interval          5
    Mem_Buf_Limit             50MB
    Skip_Long_Lines           On
    Skip_Empty_Lines          On
    DB                        /var/log/flb-storage/tail-app.db
    DB.sync                   normal
    storage.type              filesystem

# Tail Container Logs
[INPUT]
    Name                      tail
    Path                      /var/log/containers/*.log
    Parser                    docker
    Tag                       kube.*
    Refresh_Interval          5
    Mem_Buf_Limit             100MB
    DB                        /var/log/flb-storage/tail-kube.db
    storage.type              filesystem

# Systemd Logs
[INPUT]
    Name                      systemd
    Tag                       systemd.*
    Read_From_Tail            On
    Strip_Underscores         On

# Kubernetes Metadata Enrichment
[FILTER]
    Name                      kubernetes
    Match                     kube.*
    Kube_URL                  https://kubernetes.default.svc:443
    Kube_CA_File              /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    Kube_Token_File           /var/run/secrets/kubernetes.io/serviceaccount/token
    Kube_Tag_Prefix           kube.var.log.containers.
    Merge_Log                 On
    Keep_Log                  Off
    K8S-Logging.Parser        On
    K8S-Logging.Exclude       On
    Labels                    On
    Annotations               On

# Add Hostname
[FILTER]
    Name                      record_modifier
    Match                     *
    Record                    hostname ${HOSTNAME}
    Record                    environment ${ENVIRONMENT}
    Record                    cluster ${CLUSTER_NAME}

# Parse JSON Logs
[FILTER]
    Name                      parser
    Match                     app.*
    Key_Name                  log
    Parser                    json
    Reserve_Data              On
    Preserve_Key              Off

# Modify Records
[FILTER]
    Name                      modify
    Match                     *
    Remove                    _p
    Remove                    stream
    Rename                    time timestamp

# Throttle High-Volume Logs
[FILTER]
    Name                      throttle
    Match                     app.noisy-service.*
    Rate                      1000
    Window                    5
    Print_Status              True
    Interval                  30s

# Nest Kubernetes Fields
[FILTER]
    Name                      nest
    Match                     kube.*
    Operation                 nest
    Wildcard                  kubernetes_*
    Nest_under                kubernetes
    Remove_prefix             kubernetes_

# Forward to Fluentd Aggregators
[OUTPUT]
    Name                      forward
    Match                     *
    Host                      fluentd-aggregator.logging.svc.cluster.local
    Port                      24224
    Time_as_Integer           On
    Retry_Limit               5
    storage.total_limit_size  10G

# Fallback to Local File
[OUTPUT]
    Name                      file
    Match                     *
    Path                      /var/log/fluent-bit-fallback
    Format                    json_lines
```

### Fluent Bit Parsers

```ini
# parsers.conf
[PARSER]
    Name                      json
    Format                    json
    Time_Key                  time
    Time_Format               %Y-%m-%dT%H:%M:%S.%L%z
    Time_Keep                 On
    Decode_Field_As           escaped_utf8 log

[PARSER]
    Name                      docker
    Format                    json
    Time_Key                  time
    Time_Format               %Y-%m-%dT%H:%M:%S.%L%z
    Time_Keep                 On
    Decode_Field_As           json log

[PARSER]
    Name                      nginx
    Format                    regex
    Regex                     ^(?<remote>[^ ]*) (?<host>[^ ]*) (?<user>[^ ]*) \[(?<time>[^\]]*)\] "(?<method>\S+)(?: +(?<path>[^\"]*?)(?: +\S*)?)?" (?<code>[^ ]*) (?<size>[^ ]*)(?: "(?<referer>[^\"]*)" "(?<agent>[^\"]*)")?$
    Time_Key                  time
    Time_Format               %d/%b/%Y:%H:%M:%S %z

[PARSER]
    Name                      syslog
    Format                    regex
    Regex                     ^\<(?<pri>[0-9]+)\>(?<time>[^ ]* {1,2}[^ ]* [^ ]*) (?<host>[^ ]*) (?<ident>[a-zA-Z0-9_\/\.\-]*)(?:\[(?<pid>[0-9]+)\])?(?:[^\:]*\:)? *(?<message>.*)$
    Time_Key                  time
    Time_Format               %b %d %H:%M:%S

[PARSER]
    Name                      apache
    Format                    regex
    Regex                     ^(?<host>[^ ]*) [^ ]* (?<user>[^ ]*) \[(?<time>[^\]]*)\] "(?<method>\S+)(?: +(?<path>[^ ]*) +\S*)?" (?<code>[^ ]*) (?<size>[^ ]*)(?: "(?<referer>[^\"]*)" "(?<agent>[^\"]*)")?$
    Time_Key                  time
    Time_Format               %d/%b/%Y:%H:%M:%S %z
```

### Fluentd Configuration

```ruby
# fluentd.conf - Aggregation Layer
<system>
  log_level info
  workers 4
  root_dir /var/log/fluentd
</system>

# Receive from Fluent Bit
<source>
  @type forward
  @id forward_input
  port 24224
  bind 0.0.0.0

  <transport tls>
    cert_path /etc/fluentd/certs/server.crt
    private_key_path /etc/fluentd/certs/server.key
    client_cert_auth true
    ca_path /etc/fluentd/certs/ca.crt
  </transport>

  <security>
    self_hostname "#{ENV['HOSTNAME']}"
    shared_key "#{ENV['FLUENTD_SHARED_KEY']}"
  </security>
</source>

# HTTP Input for Direct Logs
<source>
  @type http
  @id http_input
  port 8888
  bind 0.0.0.0
  body_size_limit 32m
  keepalive_timeout 10s
  add_http_headers true
  add_remote_addr true

  <parse>
    @type json
  </parse>
</source>

# Tail Fluentd's Own Logs
<source>
  @type tail
  @id fluentd_logs
  path /var/log/fluentd/fluentd.log
  pos_file /var/log/fluentd/fluentd.log.pos
  tag fluentd.logs

  <parse>
    @type regexp
    expression /^(?<time>[^ ]+ [^ ]+) (?<level>[^ ]+) (?<message>.*)$/
    time_format %Y-%m-%d %H:%M:%S %z
  </parse>
</source>

# Parse and Enrich
<filter app.**>
  @type parser
  key_name log
  reserve_data true
  remove_key_name_field false
  emit_invalid_record_to_error false

  <parse>
    @type json
  </parse>
</filter>

# Extract Error Stack Traces
<filter app.**>
  @type parser
  key_name stack_trace
  reserve_data true

  <parse>
    @type multiline
    format_firstline /^[A-Z][a-zA-Z]+Error:/
    format1 /^(?<error_type>[A-Z][a-zA-Z]+Error): (?<error_message>.+)$/
    format2 /^\s+at (?<stack_line>.+)$/
  </parse>
</filter>

# GeoIP Enrichment
<filter **>
  @type geoip
  geoip_lookup_keys client_ip

  <record>
    location ${city.names.en["client_ip"]}
    country ${country.iso_code["client_ip"]}
    latitude ${location.latitude["client_ip"]}
    longitude ${location.longitude["client_ip"]}
  </record>

  skip_adding_null_record true
</filter>

# Add Processing Timestamp
<filter **>
  @type record_transformer
  enable_ruby true

  <record>
    processed_at ${time.strftime('%Y-%m-%dT%H:%M:%S.%LZ')}
    fluentd_host "#{Socket.gethostname}"
    log_id ${require 'securerandom'; SecureRandom.uuid}
  </record>
</filter>

# Detect Patterns
<filter app.**>
  @type detect_exceptions
  languages java, python, ruby, nodejs
  multiline_flush_interval 1s
  max_bytes 500000
  max_lines 1000
</filter>

# Prometheus Metrics
<filter **>
  @type prometheus

  <metric>
    name fluentd_records_total
    type counter
    desc The total number of records
    <labels>
      tag ${tag}
      hostname ${hostname}
    </labels>
  </metric>

  <metric>
    name fluentd_record_bytes_total
    type counter
    desc The total bytes of records
    key bytes
    <labels>
      tag ${tag}
    </labels>
  </metric>
</filter>

# Route by Log Level
<match app.{error,critical,fatal}>
  @type copy

  # Send to Elasticsearch
  <store>
    @type elasticsearch
    @id elasticsearch_errors
    host elasticsearch-master.logging.svc.cluster.local
    port 9200
    scheme https
    ssl_verify true
    ca_file /etc/fluentd/certs/ca.crt

    user "#{ENV['ES_USER']}"
    password "#{ENV['ES_PASSWORD']}"

    index_name error-logs-%Y.%m.%d
    type_name _doc

    logstash_format true
    logstash_prefix error-logs
    logstash_dateformat %Y.%m.%d

    include_tag_key true
    tag_key @log_name

    reconnect_on_error true
    reload_connections false
    reload_on_failure true

    request_timeout 15s

    <buffer>
      @type file
      path /var/log/fluentd/buffer/elasticsearch-errors
      flush_mode interval
      flush_interval 10s
      flush_thread_count 4
      overflow_action block
      retry_type exponential_backoff
      retry_timeout 1h
      retry_max_interval 30s
      chunk_limit_size 10M
      queue_limit_length 32
      total_limit_size 10G
    </buffer>
  </store>

  # Send to PagerDuty
  <store>
    @type pagerduty
    service_key "#{ENV['PAGERDUTY_KEY']}"

    <buffer>
      @type memory
      flush_interval 1s
    </buffer>
  </store>
</match>

# Route Application Logs
<match app.**>
  @type elasticsearch
  @id elasticsearch_app
  host elasticsearch-master.logging.svc.cluster.local
  port 9200
  scheme https
  ssl_verify true
  ca_file /etc/fluentd/certs/ca.crt

  user "#{ENV['ES_USER']}"
  password "#{ENV['ES_PASSWORD']}"

  index_name app-logs-%Y.%m.%d
  type_name _doc

  logstash_format true
  logstash_prefix app-logs
  logstash_dateformat %Y.%m.%d

  template_name app-logs
  template_file /etc/fluentd/templates/app-logs.json
  template_overwrite true

  include_tag_key true
  tag_key @log_name

  time_key @timestamp
  time_key_format %Y-%m-%dT%H:%M:%S.%L%z

  reconnect_on_error true
  reload_connections false
  reload_on_failure true

  request_timeout 15s

  <buffer>
    @type file
    path /var/log/fluentd/buffer/elasticsearch-app
    flush_mode interval
    flush_interval 30s
    flush_thread_count 8
    overflow_action block
    retry_type exponential_backoff
    retry_timeout 1h
    retry_max_interval 30s
    chunk_limit_size 20M
    queue_limit_length 64
    total_limit_size 20G
  </buffer>
</match>

# Route System Logs
<match systemd.**>
  @type elasticsearch
  @id elasticsearch_system
  host elasticsearch-master.logging.svc.cluster.local
  port 9200
  scheme https
  ssl_verify true

  user "#{ENV['ES_USER']}"
  password "#{ENV['ES_PASSWORD']}"

  index_name system-logs-%Y.%m
  logstash_format true
  logstash_prefix system-logs
  logstash_dateformat %Y.%m

  <buffer>
    @type file
    path /var/log/fluentd/buffer/elasticsearch-system
    flush_interval 60s
    chunk_limit_size 10M
  </buffer>
</match>

# Archive to S3
<match **>
  @type s3
  @id s3_archive

  aws_key_id "#{ENV['AWS_ACCESS_KEY_ID']}"
  aws_sec_key "#{ENV['AWS_SECRET_ACCESS_KEY']}"
  s3_bucket log-archive-bucket
  s3_region us-east-1

  path logs/%Y/%m/%d/

  time_slice_format %Y%m%d

  <buffer time,tag>
    @type file
    path /var/log/fluentd/buffer/s3
    timekey 3600
    timekey_wait 10m
    timekey_use_utc true
    chunk_limit_size 256m
  </buffer>

  <format>
    @type json
  </format>

  <compress>
    @type gzip
  </compress>

  store_as gzip
</match>
```

### Kubernetes Deployment

```yaml
# fluentd-deployment.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: logging
data:
  fluent.conf: |
    # Configuration content here

---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluent-bit
  namespace: logging
  labels:
    app: fluent-bit
spec:
  selector:
    matchLabels:
      app: fluent-bit
  template:
    metadata:
      labels:
        app: fluent-bit
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "2020"
        prometheus.io/path: "/api/v1/metrics/prometheus"
    spec:
      serviceAccountName: fluent-bit
      tolerations:
        - key: node-role.kubernetes.io/master
          operator: Exists
          effect: NoSchedule
      containers:
        - name: fluent-bit
          image: fluent/fluent-bit:2.1.8
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 2020
              name: metrics
              protocol: TCP
          resources:
            limits:
              cpu: 200m
              memory: 256Mi
            requests:
              cpu: 100m
              memory: 128Mi
          volumeMounts:
            - name: varlog
              mountPath: /var/log
            - name: varlibdockercontainers
              mountPath: /var/lib/docker/containers
              readOnly: true
            - name: fluent-bit-config
              mountPath: /fluent-bit/etc/
            - name: flb-storage
              mountPath: /var/log/flb-storage/
          env:
            - name: HOSTNAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
            - name: ENVIRONMENT
              value: "production"
            - name: CLUSTER_NAME
              value: "main-cluster"
      volumes:
        - name: varlog
          hostPath:
            path: /var/log
        - name: varlibdockercontainers
          hostPath:
            path: /var/lib/docker/containers
        - name: fluent-bit-config
          configMap:
            name: fluent-bit-config
        - name: flb-storage
          hostPath:
            path: /var/log/fluent-bit-storage
            type: DirectoryOrCreate

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: fluentd
  namespace: logging
  labels:
    app: fluentd
spec:
  serviceName: fluentd
  replicas: 3
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "24231"
    spec:
      serviceAccountName: fluentd
      containers:
        - name: fluentd
          image: fluent/fluentd:v1.16-1
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 24224
              name: forward
              protocol: TCP
            - containerPort: 8888
              name: http
              protocol: TCP
            - containerPort: 24231
              name: metrics
              protocol: TCP
          resources:
            limits:
              cpu: 2000m
              memory: 4Gi
            requests:
              cpu: 1000m
              memory: 2Gi
          volumeMounts:
            - name: fluentd-config
              mountPath: /fluentd/etc
            - name: buffer
              mountPath: /var/log/fluentd
            - name: certs
              mountPath: /etc/fluentd/certs
              readOnly: true
          env:
            - name: HOSTNAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: ES_USER
              valueFrom:
                secretKeyRef:
                  name: elasticsearch-credentials
                  key: username
            - name: ES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: elasticsearch-credentials
                  key: password
            - name: FLUENTD_SHARED_KEY
              valueFrom:
                secretKeyRef:
                  name: fluentd-shared-key
                  key: shared_key
      volumes:
        - name: fluentd-config
          configMap:
            name: fluentd-config
        - name: certs
          secret:
            secretName: fluentd-tls
  volumeClaimTemplates:
    - metadata:
        name: buffer
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 50Gi
```

---

## Elasticsearch Configuration

### Index Templates

```json
{
  "index_patterns": ["app-logs-*"],
  "version": 1,
  "priority": 100,
  "template": {
    "settings": {
      "number_of_shards": 3,
      "number_of_replicas": 2,
      "refresh_interval": "30s",
      "index.codec": "best_compression",
      "index.mapping.total_fields.limit": 2000,
      "index.mapping.depth.limit": 20,
      "index.mapping.nested_fields.limit": 100,
      "index.max_result_window": 10000,
      "index.max_inner_result_window": 100,
      "index.max_terms_count": 65536,
      "index.lifecycle.name": "app-logs-policy",
      "index.lifecycle.rollover_alias": "app-logs",
      "analysis": {
        "analyzer": {
          "path_analyzer": {
            "type": "custom",
            "tokenizer": "path_hierarchy",
            "filter": ["lowercase"]
          },
          "trace_id_analyzer": {
            "type": "custom",
            "tokenizer": "keyword",
            "filter": ["lowercase"]
          }
        }
      }
    },
    "mappings": {
      "dynamic_templates": [
        {
          "strings_as_keywords": {
            "match_mapping_type": "string",
            "match": "*_id",
            "mapping": {
              "type": "keyword",
              "ignore_above": 256
            }
          }
        },
        {
          "kubernetes_labels": {
            "path_match": "kubernetes.labels.*",
            "mapping": {
              "type": "keyword",
              "ignore_above": 256
            }
          }
        },
        {
          "timestamps": {
            "match_mapping_type": "string",
            "match": "*_at",
            "mapping": {
              "type": "date",
              "format": "strict_date_optional_time||epoch_millis"
            }
          }
        }
      ],
      "properties": {
        "@timestamp": {
          "type": "date",
          "format": "strict_date_optional_time||epoch_millis"
        },
        "level": {
          "type": "keyword"
        },
        "message": {
          "type": "text",
          "fields": {
            "keyword": {
              "type": "keyword",
              "ignore_above": 512
            }
          },
          "analyzer": "standard"
        },
        "trace_id": {
          "type": "keyword",
          "analyzer": "trace_id_analyzer"
        },
        "span_id": {
          "type": "keyword"
        },
        "request_id": {
          "type": "keyword"
        },
        "user_id": {
          "type": "keyword"
        },
        "service": {
          "type": "keyword"
        },
        "environment": {
          "type": "keyword"
        },
        "hostname": {
          "type": "keyword"
        },
        "kubernetes": {
          "properties": {
            "namespace": {
              "type": "keyword"
            },
            "pod_name": {
              "type": "keyword"
            },
            "container_name": {
              "type": "keyword"
            },
            "labels": {
              "type": "object",
              "dynamic": true
            },
            "annotations": {
              "type": "object",
              "dynamic": true
            }
          }
        },
        "http": {
          "properties": {
            "method": {
              "type": "keyword"
            },
            "path": {
              "type": "text",
              "fields": {
                "keyword": {
                  "type": "keyword",
                  "ignore_above": 512
                },
                "tree": {
                  "type": "text",
                  "analyzer": "path_analyzer"
                }
              }
            },
            "status_code": {
              "type": "short"
            },
            "response_time_ms": {
              "type": "float"
            },
            "request_size_bytes": {
              "type": "long"
            },
            "response_size_bytes": {
              "type": "long"
            },
            "user_agent": {
              "type": "text",
              "fields": {
                "keyword": {
                  "type": "keyword",
                  "ignore_above": 512
                }
              }
            }
          }
        },
        "error": {
          "properties": {
            "type": {
              "type": "keyword"
            },
            "message": {
              "type": "text"
            },
            "stack_trace": {
              "type": "text",
              "index": false
            }
          }
        },
        "location": {
          "properties": {
            "city": {
              "type": "keyword"
            },
            "country": {
              "type": "keyword"
            },
            "coordinates": {
              "type": "geo_point"
            }
          }
        },
        "duration_ms": {
          "type": "float"
        },
        "bytes": {
          "type": "long"
        },
        "labels": {
          "type": "object",
          "dynamic": true
        }
      }
    },
    "aliases": {
      "app-logs": {}
    }
  }
}
```

### Index Lifecycle Management

```json
{
  "policy": {
    "phases": {
      "hot": {
        "min_age": "0ms",
        "actions": {
          "rollover": {
            "max_primary_shard_size": "50gb",
            "max_age": "1d",
            "max_docs": 100000000
          },
          "set_priority": {
            "priority": 100
          }
        }
      },
      "warm": {
        "min_age": "3d",
        "actions": {
          "shrink": {
            "number_of_shards": 1
          },
          "forcemerge": {
            "max_num_segments": 1
          },
          "allocate": {
            "number_of_replicas": 1,
            "require": {
              "data": "warm"
            }
          },
          "set_priority": {
            "priority": 50
          }
        }
      },
      "cold": {
        "min_age": "30d",
        "actions": {
          "searchable_snapshot": {
            "snapshot_repository": "s3-repository"
          },
          "allocate": {
            "number_of_replicas": 0,
            "require": {
              "data": "cold"
            }
          },
          "set_priority": {
            "priority": 0
          }
        }
      },
      "delete": {
        "min_age": "90d",
        "actions": {
          "delete": {}
        }
      }
    }
  }
}
```

### Elasticsearch Configuration

```yaml
# elasticsearch.yml
cluster.name: logging-cluster
node.name: ${HOSTNAME}
node.roles: [master, data_hot, data_content]

path.data: /usr/share/elasticsearch/data
path.logs: /usr/share/elasticsearch/logs

network.host: 0.0.0.0
http.port: 9200
transport.port: 9300

discovery.seed_hosts:
  - elasticsearch-0.elasticsearch-headless
  - elasticsearch-1.elasticsearch-headless
  - elasticsearch-2.elasticsearch-headless

cluster.initial_master_nodes:
  - elasticsearch-0
  - elasticsearch-1
  - elasticsearch-2

xpack.security.enabled: true
xpack.security.transport.ssl.enabled: true
xpack.security.transport.ssl.verification_mode: certificate
xpack.security.transport.ssl.keystore.path: certs/elastic-certificates.p12
xpack.security.transport.ssl.truststore.path: certs/elastic-certificates.p12

xpack.security.http.ssl.enabled: true
xpack.security.http.ssl.keystore.path: certs/elastic-certificates.p12
xpack.security.http.ssl.truststore.path: certs/elastic-certificates.p12

xpack.monitoring.collection.enabled: true

indices.memory.index_buffer_size: 20%
indices.queries.cache.size: 10%
indices.requests.cache.size: 2%

thread_pool.write.queue_size: 1000
thread_pool.search.queue_size: 1000

bootstrap.memory_lock: true

action.destructive_requires_name: true
```

---

## Kibana Setup

### Kibana Configuration

```yaml
# kibana.yml
server.name: kibana
server.host: "0.0.0.0"
server.port: 5601

elasticsearch.hosts: ["https://elasticsearch-master:9200"]
elasticsearch.username: "kibana_system"
elasticsearch.password: "${KIBANA_PASSWORD}"

elasticsearch.ssl.certificateAuthorities: ["/usr/share/kibana/config/certs/ca.crt"]
elasticsearch.ssl.verificationMode: full

xpack.security.enabled: true
xpack.security.encryptionKey: "${ENCRYPTION_KEY}"
xpack.security.session.idleTimeout: "1h"
xpack.security.session.lifespan: "30d"

xpack.encryptedSavedObjects.encryptionKey: "${ENCRYPTED_SAVED_OBJECTS_KEY}"

logging.root.level: info

monitoring.enabled: true
monitoring.kibana.collection.enabled: true

telemetry.enabled: false
telemetry.optIn: false
```

### Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Application Logs Overview",
    "panels": [
      {
        "id": "log-volume-timeline",
        "type": "visualization",
        "visualization": {
          "type": "line",
          "query": {
            "query": "level:* AND service:*",
            "language": "lucene"
          },
          "aggregations": {
            "date_histogram": {
              "field": "@timestamp",
              "interval": "1m",
              "min_doc_count": 0
            },
            "terms": {
              "field": "level",
              "size": 10
            }
          }
        },
        "title": "Log Volume by Level"
      },
      {
        "id": "error-rate",
        "type": "metric",
        "visualization": {
          "type": "gauge",
          "query": {
            "query": "level:(ERROR OR FATAL OR CRITICAL)",
            "language": "lucene"
          },
          "metric": {
            "type": "count"
          }
        },
        "title": "Error Rate"
      },
      {
        "id": "top-services",
        "type": "visualization",
        "visualization": {
          "type": "pie",
          "aggregations": {
            "terms": {
              "field": "service",
              "size": 10
            }
          }
        },
        "title": "Top Services by Log Volume"
      },
      {
        "id": "response-time-percentiles",
        "type": "visualization",
        "visualization": {
          "type": "area",
          "aggregations": {
            "percentiles": {
              "field": "http.response_time_ms",
              "percents": [50, 90, 95, 99]
            }
          }
        },
        "title": "Response Time Percentiles"
      },
      {
        "id": "error-breakdown",
        "type": "table",
        "query": {
          "query": "level:(ERROR OR FATAL)",
          "language": "lucene"
        },
        "columns": [
          "@timestamp",
          "service",
          "error.type",
          "error.message",
          "trace_id"
        ],
        "title": "Recent Errors"
      }
    ],
    "filters": [
      {
        "meta": {
          "index": "app-logs-*",
          "type": "time",
          "key": "@timestamp",
          "params": {}
        },
        "range": {
          "@timestamp": {
            "gte": "now-24h",
            "lte": "now"
          }
        }
      }
    ],
    "refresh_interval": "30s"
  }
}
```

### Saved Searches

```json
{
  "searches": [
    {
      "title": "Application Errors",
      "query": {
        "query": "level:(ERROR OR FATAL OR CRITICAL)",
        "language": "lucene"
      },
      "columns": [
        "@timestamp",
        "service",
        "message",
        "error.type",
        "trace_id"
      ],
      "sort": [["@timestamp", "desc"]]
    },
    {
      "title": "Slow Requests",
      "query": {
        "query": "http.response_time_ms:>1000",
        "language": "lucene"
      },
      "columns": [
        "@timestamp",
        "service",
        "http.method",
        "http.path",
        "http.response_time_ms",
        "trace_id"
      ],
      "sort": [["http.response_time_ms", "desc"]]
    },
    {
      "title": "Failed Requests",
      "query": {
        "query": "http.status_code:>=500",
        "language": "lucene"
      },
      "columns": [
        "@timestamp",
        "service",
        "http.method",
        "http.path",
        "http.status_code",
        "trace_id"
      ]
    }
  ]
}
```

---

## Structured Logging Libraries

### Python Logging

```python
# structured_logging.py
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
import traceback
import uuid

class StructuredLogger:
    """Production-grade structured logger for Python applications."""

    def __init__(
        self,
        name: str,
        service: str,
        environment: str,
        level: int = logging.INFO,
        add_trace_context: bool = True
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.service = service
        self.environment = environment
        self.add_trace_context = add_trace_context

        # Configure handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter(service, environment))
        self.logger.addHandler(handler)

    def _log(
        self,
        level: str,
        message: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Internal logging method with structured context."""
        extra = {
            'service': self.service,
            'environment': self.environment,
            'trace_id': trace_id or self._get_trace_id(),
            'span_id': span_id or self._get_span_id(),
            **kwargs
        }

        log_method = getattr(self.logger, level.lower())
        log_method(message, extra=extra)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info level message."""
        self._log('INFO', message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning level message."""
        self._log('WARNING', message, **kwargs)

    def error(
        self,
        message: str,
        exc_info: Optional[Exception] = None,
        **kwargs: Any
    ) -> None:
        """Log error level message with optional exception."""
        if exc_info:
            kwargs['error'] = {
                'type': type(exc_info).__name__,
                'message': str(exc_info),
                'stack_trace': traceback.format_exc()
            }
        self._log('ERROR', message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical level message."""
        self._log('CRITICAL', message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug level message."""
        self._log('DEBUG', message, **kwargs)

    def _get_trace_id(self) -> Optional[str]:
        """Get trace ID from current context."""
        if not self.add_trace_context:
            return None

        # Import here to avoid circular dependencies
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span:
                return format(span.get_span_context().trace_id, '032x')
        except ImportError:
            pass

        return None

    def _get_span_id(self) -> Optional[str]:
        """Get span ID from current context."""
        if not self.add_trace_context:
            return None

        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span:
                return format(span.get_span_context().span_id, '016x')
        except ImportError:
            pass

        return None


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, service: str, environment: str):
        super().__init__()
        self.service = service
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            '@timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'service': self.service,
            'environment': self.environment,
            'process_id': record.process,
            'thread_id': record.thread,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields
        if hasattr(record, 'trace_id') and record.trace_id:
            log_data['trace_id'] = record.trace_id

        if hasattr(record, 'span_id') and record.span_id:
            log_data['span_id'] = record.span_id

        # Add any additional fields from extra
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename',
                          'funcName', 'levelname', 'levelno', 'lineno',
                          'module', 'msecs', 'message', 'pathname', 'process',
                          'processName', 'relativeCreated', 'thread',
                          'threadName', 'exc_info', 'exc_text', 'stack_info']:
                if not key.startswith('_'):
                    log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data['error'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'stack_trace': self.formatException(record.exc_info)
            }

        return json.dumps(log_data, default=str)


# Usage example
logger = StructuredLogger(
    name='my-app',
    service='user-service',
    environment='production'
)

# Simple logging
logger.info('User logged in', user_id='12345', session_id='abc-def')

# Error logging with exception
try:
    result = 1 / 0
except ZeroDivisionError as e:
    logger.error('Division error occurred', exc_info=e, operation='divide')

# HTTP request logging
logger.info(
    'HTTP request processed',
    http={
        'method': 'GET',
        'path': '/api/users',
        'status_code': 200,
        'response_time_ms': 45.2
    },
    user_id='12345'
)
```

### Node.js Winston

```javascript
// structured-logger.js
const winston = require('winston');
const { trace, context } = require('@opentelemetry/api');

class StructuredLogger {
  constructor(options = {}) {
    const {
      service,
      environment,
      level = 'info',
      addTraceContext = true
    } = options;

    this.service = service;
    this.environment = environment;
    this.addTraceContext = addTraceContext;

    const customFormat = winston.format.combine(
      winston.format.timestamp({ format: 'YYYY-MM-DDTHH:mm:ss.SSSZ' }),
      winston.format.errors({ stack: true }),
      winston.format.printf((info) => this.formatLog(info))
    );

    this.logger = winston.createLogger({
      level,
      format: customFormat,
      defaultMeta: {
        service: this.service,
        environment: this.environment
      },
      transports: [
        new winston.transports.Console({
          handleExceptions: true,
          handleRejections: true
        })
      ],
      exitOnError: false
    });
  }

  formatLog(info) {
    const logData = {
      '@timestamp': info.timestamp,
      level: info.level.toUpperCase(),
      message: info.message,
      service: this.service,
      environment: this.environment,
      ...info
    };

    // Remove winston metadata
    delete logData.timestamp;
    delete logData.level;

    // Add trace context
    if (this.addTraceContext) {
      const span = trace.getActiveSpan();
      if (span) {
        const spanContext = span.spanContext();
        logData.trace_id = spanContext.traceId;
        logData.span_id = spanContext.spanId;
      }
    }

    // Format error if present
    if (info.error && info.error instanceof Error) {
      logData.error = {
        type: info.error.name,
        message: info.error.message,
        stack_trace: info.error.stack
      };
      delete logData.error;
    }

    return JSON.stringify(logData);
  }

  info(message, meta = {}) {
    this.logger.info(message, meta);
  }

  warn(message, meta = {}) {
    this.logger.warn(message, meta);
  }

  error(message, error, meta = {}) {
    this.logger.error(message, {
      ...meta,
      error: error instanceof Error ? error : new Error(String(error))
    });
  }

  debug(message, meta = {}) {
    this.logger.debug(message, meta);
  }

  http(message, meta = {}) {
    this.logger.http(message, meta);
  }
}

// Express middleware for HTTP logging
function httpLoggingMiddleware(logger) {
  return (req, res, next) => {
    const start = Date.now();

    res.on('finish', () => {
      const duration = Date.now() - start;

      logger.http('HTTP request processed', {
        http: {
          method: req.method,
          path: req.path,
          status_code: res.statusCode,
          response_time_ms: duration,
          request_size_bytes: req.headers['content-length'] || 0,
          response_size_bytes: res.get('content-length') || 0,
          user_agent: req.get('user-agent')
        },
        request_id: req.id,
        user_id: req.user?.id,
        client_ip: req.ip
      });
    });

    next();
  };
}

module.exports = { StructuredLogger, httpLoggingMiddleware };

// Usage
const logger = new StructuredLogger({
  service: 'user-service',
  environment: 'production'
});

logger.info('User logged in', {
  user_id: '12345',
  session_id: 'abc-def'
});

logger.error('Database connection failed', new Error('Connection timeout'), {
  database: 'users',
  retry_count: 3
});
```

### Go Zap

```go
// structured_logger.go
package logging

import (
	"context"
	"os"

	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

type StructuredLogger struct {
	logger      *zap.Logger
	service     string
	environment string
}

func NewStructuredLogger(service, environment, level string) (*StructuredLogger, error) {
	config := zap.NewProductionConfig()
	config.EncoderConfig.TimeKey = "@timestamp"
	config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	config.EncoderConfig.MessageKey = "message"
	config.EncoderConfig.LevelKey = "level"
	config.EncoderConfig.EncodeLevel = zapcore.CapitalLevelEncoder

	// Set log level
	var zapLevel zapcore.Level
	if err := zapLevel.UnmarshalText([]byte(level)); err != nil {
		zapLevel = zapcore.InfoLevel
	}
	config.Level = zap.NewAtomicLevelAt(zapLevel)

	// Build logger
	logger, err := config.Build(
		zap.AddCaller(),
		zap.AddStacktrace(zapcore.ErrorLevel),
		zap.Fields(
			zap.String("service", service),
			zap.String("environment", environment),
			zap.Int("pid", os.Getpid()),
		),
	)
	if err != nil {
		return nil, err
	}

	return &StructuredLogger{
		logger:      logger,
		service:     service,
		environment: environment,
	}, nil
}

func (l *StructuredLogger) WithTraceContext(ctx context.Context) *zap.Logger {
	span := trace.SpanFromContext(ctx)
	if !span.IsRecording() {
		return l.logger
	}

	spanContext := span.SpanContext()
	return l.logger.With(
		zap.String("trace_id", spanContext.TraceID().String()),
		zap.String("span_id", spanContext.SpanID().String()),
	)
}

func (l *StructuredLogger) Info(ctx context.Context, msg string, fields ...zap.Field) {
	l.WithTraceContext(ctx).Info(msg, fields...)
}

func (l *StructuredLogger) Warn(ctx context.Context, msg string, fields ...zap.Field) {
	l.WithTraceContext(ctx).Warn(msg, fields...)
}

func (l *StructuredLogger) Error(ctx context.Context, msg string, err error, fields ...zap.Field) {
	allFields := append(fields, zap.Error(err))
	l.WithTraceContext(ctx).Error(msg, allFields...)
}

func (l *StructuredLogger) Debug(ctx context.Context, msg string, fields ...zap.Field) {
	l.WithTraceContext(ctx).Debug(msg, fields...)
}

func (l *StructuredLogger) Fatal(ctx context.Context, msg string, fields ...zap.Field) {
	l.WithTraceContext(ctx).Fatal(msg, fields...)
}

func (l *StructuredLogger) Sync() error {
	return l.logger.Sync()
}

// HTTP middleware for Go
func (l *StructuredLogger) HTTPMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Wrap response writer to capture status code
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(wrapped, r)

		duration := time.Since(start)

		l.Info(r.Context(), "HTTP request processed",
			zap.String("http.method", r.Method),
			zap.String("http.path", r.URL.Path),
			zap.Int("http.status_code", wrapped.statusCode),
			zap.Float64("http.response_time_ms", float64(duration.Milliseconds())),
			zap.String("http.user_agent", r.UserAgent()),
			zap.String("client_ip", r.RemoteAddr),
		)
	})
}

type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// Usage example
func main() {
	logger, err := NewStructuredLogger("user-service", "production", "info")
	if err != nil {
		panic(err)
	}
	defer logger.Sync()

	ctx := context.Background()

	logger.Info(ctx, "User logged in",
		zap.String("user_id", "12345"),
		zap.String("session_id", "abc-def"),
	)

	if err := someOperation(); err != nil {
		logger.Error(ctx, "Operation failed", err,
			zap.String("operation", "database_query"),
			zap.Int("retry_count", 3),
		)
	}
}
```

---

## Log Correlation

### Trace ID Propagation

```python
# trace_correlation.py
import uuid
from contextvars import ContextVar
from typing import Optional, Dict, Any
from functools import wraps

# Context variables for trace and request IDs
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
span_id_var: ContextVar[Optional[str]] = ContextVar('span_id', default=None)

class CorrelationContext:
    """Manages correlation IDs for distributed tracing."""

    @staticmethod
    def set_trace_id(trace_id: str) -> None:
        """Set the current trace ID."""
        trace_id_var.set(trace_id)

    @staticmethod
    def get_trace_id() -> Optional[str]:
        """Get the current trace ID."""
        return trace_id_var.get()

    @staticmethod
    def set_request_id(request_id: str) -> None:
        """Set the current request ID."""
        request_id_var.set(request_id)

    @staticmethod
    def get_request_id() -> Optional[str]:
        """Get the current request ID."""
        return request_id_var.get()

    @staticmethod
    def set_span_id(span_id: str) -> None:
        """Set the current span ID."""
        span_id_var.set(span_id)

    @staticmethod
    def get_span_id() -> Optional[str]:
        """Get the current span ID."""
        return span_id_var.get()

    @staticmethod
    def generate_trace_id() -> str:
        """Generate a new trace ID."""
        return str(uuid.uuid4())

    @staticmethod
    def get_correlation_context() -> Dict[str, Any]:
        """Get all correlation IDs."""
        return {
            'trace_id': CorrelationContext.get_trace_id(),
            'request_id': CorrelationContext.get_request_id(),
            'span_id': CorrelationContext.get_span_id()
        }

# Flask middleware
from flask import Flask, request, g

def add_correlation_middleware(app: Flask) -> None:
    """Add correlation ID middleware to Flask app."""

    @app.before_request
    def before_request():
        # Extract or generate trace ID
        trace_id = request.headers.get('X-Trace-Id')
        if not trace_id:
            trace_id = CorrelationContext.generate_trace_id()

        # Extract or generate request ID
        request_id = request.headers.get('X-Request-Id')
        if not request_id:
            request_id = str(uuid.uuid4())

        # Set context
        CorrelationContext.set_trace_id(trace_id)
        CorrelationContext.set_request_id(request_id)

        # Store in g for easy access
        g.trace_id = trace_id
        g.request_id = request_id

    @app.after_request
    def after_request(response):
        # Add correlation IDs to response headers
        response.headers['X-Trace-Id'] = CorrelationContext.get_trace_id()
        response.headers['X-Request-Id'] = CorrelationContext.get_request_id()
        return response

# Requests client wrapper
import requests

class CorrelatedHTTPClient:
    """HTTP client that propagates correlation IDs."""

    @staticmethod
    def get(url: str, **kwargs) -> requests.Response:
        """Make GET request with correlation headers."""
        headers = kwargs.get('headers', {})
        headers.update(CorrelatedHTTPClient._get_correlation_headers())
        kwargs['headers'] = headers
        return requests.get(url, **kwargs)

    @staticmethod
    def post(url: str, **kwargs) -> requests.Response:
        """Make POST request with correlation headers."""
        headers = kwargs.get('headers', {})
        headers.update(CorrelatedHTTPClient._get_correlation_headers())
        kwargs['headers'] = headers
        return requests.post(url, **kwargs)

    @staticmethod
    def _get_correlation_headers() -> Dict[str, str]:
        """Get correlation headers from context."""
        headers = {}

        trace_id = CorrelationContext.get_trace_id()
        if trace_id:
            headers['X-Trace-Id'] = trace_id

        request_id = CorrelationContext.get_request_id()
        if request_id:
            headers['X-Request-Id'] = request_id

        return headers
```

### Kibana Queries for Correlation

```json
{
  "correlation_queries": {
    "find_by_trace_id": {
      "query": {
        "match": {
          "trace_id": "abc123-def456-ghi789"
        }
      },
      "sort": [
        { "@timestamp": "asc" }
      ]
    },
    "find_request_flow": {
      "query": {
        "bool": {
          "must": [
            { "match": { "request_id": "req-12345" } }
          ]
        }
      },
      "aggs": {
        "services": {
          "terms": {
            "field": "service",
            "order": { "_key": "asc" }
          },
          "aggs": {
            "timeline": {
              "date_histogram": {
                "field": "@timestamp",
                "interval": "1s"
              }
            }
          }
        }
      }
    },
    "error_trace": {
      "query": {
        "bool": {
          "must": [
            { "match": { "trace_id": "abc123" } },
            { "terms": { "level": ["ERROR", "FATAL", "CRITICAL"] } }
          ]
        }
      }
    }
  }
}
```

---

## Retention and Archival

### Retention Policy Strategy

```yaml
# Retention tiers
retention_strategy:
  hot_tier:
    duration: 7d
    storage: elasticsearch
    access: real_time
    cost_per_gb: 0.30
    use_cases:
      - active_debugging
      - real_time_monitoring
      - alerting

  warm_tier:
    duration: 30d
    storage: elasticsearch_searchable_snapshot
    access: fast_query
    cost_per_gb: 0.15
    use_cases:
      - historical_analysis
      - compliance_review
      - trend_analysis

  cold_tier:
    duration: 90d
    storage: s3_glacier
    access: slow_query
    cost_per_gb: 0.004
    use_cases:
      - compliance_archive
      - audit_trail
      - long_term_retention

  archive_tier:
    duration: 7y
    storage: s3_deep_archive
    access: restore_required
    cost_per_gb: 0.00099
    use_cases:
      - regulatory_compliance
      - legal_hold
```

### S3 Archive Configuration

```python
# log_archiver.py
import boto3
from datetime import datetime, timedelta
from typing import List, Dict, Any

class LogArchiver:
    """Archive old logs to S3 with lifecycle policies."""

    def __init__(
        self,
        bucket_name: str,
        region: str = 'us-east-1'
    ):
        self.s3_client = boto3.client('s3', region_name=region)
        self.bucket_name = bucket_name

    def create_lifecycle_policy(self) -> None:
        """Create S3 lifecycle policy for log retention."""
        lifecycle_config = {
            'Rules': [
                {
                    'Id': 'TransitionToIA',
                    'Status': 'Enabled',
                    'Prefix': 'logs/',
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 90,
                            'StorageClass': 'GLACIER'
                        },
                        {
                            'Days': 365,
                            'StorageClass': 'DEEP_ARCHIVE'
                        }
                    ],
                    'Expiration': {
                        'Days': 2555  # 7 years
                    }
                },
                {
                    'Id': 'DeleteIncompleteMultipartUploads',
                    'Status': 'Enabled',
                    'Prefix': 'logs/',
                    'AbortIncompleteMultipartUpload': {
                        'DaysAfterInitiation': 7
                    }
                }
            ]
        }

        self.s3_client.put_bucket_lifecycle_configuration(
            Bucket=self.bucket_name,
            LifecycleConfiguration=lifecycle_config
        )

    def enable_intelligent_tiering(self) -> None:
        """Enable S3 Intelligent-Tiering for cost optimization."""
        self.s3_client.put_bucket_intelligent_tiering_configuration(
            Bucket=self.bucket_name,
            Id='LogArchiveOptimization',
            IntelligentTieringConfiguration={
                'Id': 'LogArchiveOptimization',
                'Status': 'Enabled',
                'Tierings': [
                    {
                        'Days': 90,
                        'AccessTier': 'ARCHIVE_ACCESS'
                    },
                    {
                        'Days': 180,
                        'AccessTier': 'DEEP_ARCHIVE_ACCESS'
                    }
                ]
            }
        )
```

---

## Log-Based Metrics

### Prometheus Metrics from Logs

```yaml
# Fluentd Prometheus plugin configuration
<filter app.**>
  @type prometheus

  # Counter: Total log entries
  <metric>
    name log_entries_total
    type counter
    desc Total number of log entries
    <labels>
      service ${service}
      level ${level}
      environment ${environment}
    </labels>
  </metric>

  # Counter: HTTP requests
  <metric>
    name http_requests_total
    type counter
    desc Total HTTP requests
    <labels>
      service ${service}
      method ${http.method}
      status ${http.status_code}
      path ${http.path}
    </labels>
  </metric>

  # Histogram: Response times
  <metric>
    name http_response_time_seconds
    type histogram
    desc HTTP response time distribution
    key http.response_time_ms
    buckets 0.01,0.05,0.1,0.5,1.0,2.5,5.0,10.0
    <labels>
      service ${service}
      method ${http.method}
    </labels>
  </metric>

  # Gauge: Active users
  <metric>
    name active_users
    type gauge
    desc Number of active users
    key user_count
    <labels>
      service ${service}
    </labels>
  </metric>
</filter>
```

### mtail Configuration

```go
// http_metrics.mtail
counter http_requests_total by service, method, status
histogram http_response_time by service, method buckets 10, 50, 100, 500, 1000, 5000

/^{"@timestamp":"(?P<timestamp>[^"]+)".*"service":"(?P<service>[^"]+)".*"http":{"method":"(?P<method>[^"]+)".*"status_code":(?P<status>\d+).*"response_time_ms":(?P<response_time>[\d.]+)/ {
  http_requests_total[$service][$method][$status]++
  http_response_time[$service][$method] = $response_time
}
```

---

## ELK vs Loki Comparison

### Architecture Comparison

```yaml
elk_stack:
  components:
    - elasticsearch
    - logstash_or_fluentd
    - kibana

  strengths:
    - full_text_search
    - rich_query_language
    - powerful_aggregations
    - mature_ecosystem
    - advanced_analytics

  weaknesses:
    - high_resource_usage
    - expensive_at_scale
    - complex_operations
    - slower_ingestion

  best_for:
    - full_text_search_required
    - complex_analytics
    - unstructured_logs
    - compliance_requirements

loki:
  components:
    - loki
    - promtail
    - grafana

  strengths:
    - cost_effective
    - simple_operations
    - fast_ingestion
    - label_based_indexing
    - prometheus_integration

  weaknesses:
    - limited_search
    - no_full_text_index
    - newer_ecosystem
    - basic_aggregations

  best_for:
    - kubernetes_logs
    - metrics_correlation
    - cost_optimization
    - label_based_queries
```

### Cost Analysis

```yaml
# Monthly cost comparison (10TB/month)
cost_comparison:
  elk_stack:
    elasticsearch_compute: 3000
    elasticsearch_storage: 3000
    data_transfer: 500
    total_monthly: 6500

  loki:
    loki_compute: 800
    object_storage: 400
    data_transfer: 300
    total_monthly: 1500

  savings:
    percentage: 77
    annual_savings: 60000
```

---

## Performance Optimization

### Elasticsearch Optimization

```yaml
# Performance tuning
optimization_strategies:
  indexing:
    - use_bulk_api
    - increase_refresh_interval
    - disable_replicas_during_bulk
    - use_index_sorting
    - optimize_mapping

  querying:
    - use_filter_context
    - minimize_script_usage
    - use_aggregation_caching
    - optimize_shard_count
    - use_index_patterns

  storage:
    - enable_compression
    - use_forcemerge
    - implement_ilm
    - optimize_field_types
    - disable_unused_features
```

### Buffer Configuration

```ini
# Optimized Fluent Bit buffering
[OUTPUT]
    Name                      forward
    Match                     *
    Host                      fluentd
    Port                      24224

    # Buffer configuration
    storage.total_limit_size  20G
    Retry_Limit               5

    # Network optimization
    net.keepalive             on
    net.keepalive_idle_timeout 30
    net.keepalive_max_recycle  2000

    # Compression
    Compress                  gzip
```

### Query Optimization

```json
{
  "optimized_query": {
    "bool": {
      "filter": [
        { "range": { "@timestamp": { "gte": "now-1h" } } },
        { "term": { "service": "api-gateway" } }
      ],
      "must": [
        { "match": { "level": "ERROR" } }
      ]
    }
  },
  "size": 100,
  "_source": ["@timestamp", "message", "trace_id"],
  "sort": [{ "@timestamp": "desc" }],
  "track_total_hits": false
}
```

### Cost Management

```yaml
# Cost optimization strategies
cost_management:
  sampling:
    - sample_debug_logs: 10%
    - sample_info_logs: 50%
    - sample_error_logs: 100%
    - sample_noisy_services: 5%

  compression:
    - use_gzip_compression
    - compress_archived_logs
    - use_columnar_formats

  retention:
    - delete_debug_after: 3d
    - delete_info_after: 30d
    - archive_errors_after: 90d

  indexing:
    - reduce_field_count
    - disable_dynamic_mapping
    - use_keyword_not_text
    - disable_norms_on_text
```

---

## Production Checklist

### Deployment Checklist

```yaml
pre_deployment:
  - [ ] Size Elasticsearch cluster appropriately
  - [ ] Configure ILM policies
  - [ ] Set up backup and restore
  - [ ] Enable TLS encryption
  - [ ] Configure authentication
  - [ ] Set up monitoring
  - [ ] Test failover scenarios
  - [ ] Document runbooks

monitoring:
  - [ ] Monitor ingestion rate
  - [ ] Track error rates
  - [ ] Monitor storage usage
  - [ ] Alert on pipeline failures
  - [ ] Track query performance
  - [ ] Monitor cluster health

security:
  - [ ] Encrypt data in transit
  - [ ] Encrypt data at rest
  - [ ] Implement RBAC
  - [ ] Audit access logs
  - [ ] Secure API endpoints
  - [ ] Rotate credentials
  - [ ] Scan for vulnerabilities
```

This comprehensive guide provides production-ready configurations for implementing a complete log aggregation pipeline with Fluentd, Elasticsearch, and Kibana, including structured logging, correlation, retention strategies, and performance optimization.
