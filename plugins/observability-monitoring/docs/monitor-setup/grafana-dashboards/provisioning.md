# Dashboard Provisioning

Automate dashboard deployment using provisioning.

## Provisioning Configuration

**provisioning/dashboards/default.yml:**
```yaml
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: 'Services'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/dashboards
      foldersFromFilesStructure: true
```

## Dashboard File Structure

```
/etc/grafana/dashboards/
├── services/
│   ├── api-gateway.json
│   ├── user-service.json
│   └── payment-service.json
├── infrastructure/
│   ├── kubernetes.json
│   ├── nodes.json
│   └── network.json
└── business/
    ├── sales.json
    └── conversions.json
```

## ConfigMap for Kubernetes

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards
  namespace: monitoring
data:
  service-overview.json: |
    {
      "dashboard": {
        "uid": "service-overview",
        "title": "Service Overview",
        "panels": [...]
      }
    }
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-provider
  namespace: monitoring
data:
  dashboards.yml: |
    apiVersion: 1
    providers:
      - name: 'default'
        orgId: 1
        folder: ''
        type: file
        disableDeletion: false
        options:
          path: /etc/grafana/dashboards
```

## Terraform Provisioning

```hcl
resource "grafana_dashboard" "service_overview" {
  config_json = file("${path.module}/dashboards/service-overview.json")
  folder      = grafana_folder.services.id
  overwrite   = true
}

resource "grafana_folder" "services" {
  title = "Services"
}
```

## Dashboard Versioning

**Git Repository Structure:**
```
dashboards/
├── README.md
├── production/
│   ├── services/
│   │   ├── api-gateway.json
│   │   └── user-service.json
│   └── infrastructure/
│       └── kubernetes.json
├── staging/
│   └── ...
└── scripts/
    ├── deploy.sh
    └── validate.sh
```

**Validation Script:**
```bash
#!/bin/bash
# validate.sh

for dashboard in dashboards/**/*.json; do
  echo "Validating $dashboard"
  jq empty "$dashboard" || exit 1

  # Check required fields
  jq -e '.dashboard.uid' "$dashboard" > /dev/null || {
    echo "Missing UID in $dashboard"
    exit 1
  }
done

echo "All dashboards valid"
```
