# Deployment & Distribution Guide

Comprehensive deployment strategies for multi-platform applications including App Store submission, Play Store distribution, web deployment, code signing, and automated CI/CD pipelines.

## Table of Contents

1. [iOS App Store Deployment](#ios-app-store-deployment)
2. [Google Play Store Deployment](#google-play-store-deployment)
3. [Web Deployment](#web-deployment)
4. [Desktop Distribution](#desktop-distribution)
5. [CI/CD Pipelines](#cicd-pipelines)

---

## iOS App Store Deployment

### Provisioning Profiles and Code Signing

**Certificate Types:**

```
Development Certificates:
  - iOS App Development
  - Apple Development
  Use: Local development and testing

Distribution Certificates:
  - iOS Distribution
  - Apple Distribution
  Use: App Store submission and Ad Hoc distribution
```

**Provisioning Profiles:**

```bash
# Development Profile
- Type: iOS App Development
- Certificates: Development certificates
- Devices: Registered test devices
- App ID: com.example.myapp
- Capabilities: Push Notifications, In-App Purchase, etc.

# App Store Profile
- Type: App Store
- Certificates: Distribution certificate
- App ID: com.example.myapp
- Capabilities: Same as development
```

**Automated Profile Management with Fastlane:**

```ruby
# fastlane/Matchfile
git_url("https://github.com/mycompany/certificates")
storage_mode("git")
type("development")
app_identifier(["com.example.myapp"])
username("apple@example.com")
team_id("ABC123DEF4")

# fastlane/Fastfile
default_platform(:ios)

platform :ios do
  desc "Sync certificates and profiles"
  lane :sync_certificates do
    match(
      type: "development",
      readonly: is_ci
    )

    match(
      type: "appstore",
      readonly: is_ci
    )
  end

  desc "Build for App Store"
  lane :build_release do
    sync_certificates

    increment_build_number(
      xcodeproj: "MyApp.xcodeproj"
    )

    build_app(
      scheme: "MyApp",
      configuration: "Release",
      export_method: "app-store",
      export_options: {
        provisioningProfiles: {
          "com.example.myapp" => "match AppStore com.example.myapp"
        }
      }
    )
  end
end
```

### TestFlight Beta Distribution

**Fastlane Configuration:**

```ruby
# fastlane/Fastfile
platform :ios do
  desc "Upload to TestFlight"
  lane :beta do
    build_release

    upload_to_testflight(
      username: "apple@example.com",
      app_identifier: "com.example.myapp",
      skip_waiting_for_build_processing: false,
      changelog: File.read("../CHANGELOG.md"),
      distribute_external: true,
      groups: ["External Testers"],
      notify_external_testers: true,
      beta_app_feedback_email: "feedback@example.com",
      beta_app_description: "Latest beta build with new features"
    )

    # Send notification
    slack(
      message: "New TestFlight build available!",
      success: true,
      slack_url: ENV['SLACK_WEBHOOK_URL']
    )
  end
end
```

### App Store Submission

**App Store Connect Metadata:**

```yaml
# metadata/en-US/description.txt
MyApp is a powerful cross-platform application that helps you manage your daily tasks efficiently.

Features:
- User profile management
- Real-time synchronization
- Offline support
- Clean and intuitive interface
- Dark mode support

# metadata/en-US/keywords.txt
productivity,task management,sync,cross-platform

# metadata/en-US/release_notes.txt
What's New in Version 1.2.0:

- Improved profile editing experience
- Enhanced offline sync
- Bug fixes and performance improvements
- New dark mode enhancements

# metadata/en-US/privacy_url.txt
https://example.com/privacy

# metadata/en-US/support_url.txt
https://example.com/support
```

**Fastlane Metadata Upload:**

```ruby
platform :ios do
  desc "Upload metadata and screenshots"
  lane :upload_metadata do
    deliver(
      username: "apple@example.com",
      app_identifier: "com.example.myapp",
      metadata_path: "./metadata",
      screenshots_path: "./screenshots",
      skip_binary_upload: true,
      skip_screenshots: false,
      force: true
    )
  end

  desc "Submit to App Store"
  lane :release do
    build_release

    deliver(
      username: "apple@example.com",
      app_identifier: "com.example.myapp",
      submit_for_review: true,
      automatic_release: false,
      submission_information: {
        add_id_info_uses_idfa: false,
        export_compliance_uses_encryption: false
      },
      precheck_include_in_app_purchases: false
    )

    slack(
      message: "App submitted to App Store for review!",
      success: true
    )
  end
end
```

**Screenshot Generation:**

```swift
// UITests/ScreenshotTests.swift
import XCTest

final class ScreenshotTests: XCTestCase {
    var app: XCUIApplication!

    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        app = XCUIApplication()
        setupSnapshot(app)
        app.launch()
    }

    func testGenerateScreenshots() {
        // Login
        let emailField = app.textFields["email-field"]
        emailField.tap()
        emailField.typeText("demo@example.com")

        let passwordField = app.secureTextFields["password-field"]
        passwordField.tap()
        passwordField.typeText("DemoPassword123")

        app.buttons["login-button"].tap()
        sleep(2)

        // Screenshot 1: Dashboard
        snapshot("01Dashboard")

        // Screenshot 2: Profile
        app.buttons["profile-button"].tap()
        sleep(1)
        snapshot("02Profile")

        // Screenshot 3: Settings
        app.buttons["settings-button"].tap()
        sleep(1)
        snapshot("03Settings")

        // Screenshot 4: Dark Mode
        app.switches["dark-mode-toggle"].tap()
        sleep(1)
        snapshot("04DarkMode")
    }
}
```

**Fastlane Snapshot Configuration:**

```ruby
# Snapfile
devices([
  "iPhone 15 Pro Max",
  "iPhone 15 Pro",
  "iPhone 15",
  "iPhone SE (3rd generation)",
  "iPad Pro (12.9-inch) (6th generation)",
  "iPad Pro (11-inch) (4th generation)"
])

languages([
  "en-US",
  "es-ES",
  "de-DE",
  "fr-FR",
  "ja-JP",
  "zh-Hans"
])

scheme("MyApp")
output_directory("./screenshots")
clear_previous_screenshots(true)
override_status_bar(true)
```

### App Review Guidelines Checklist

```markdown
# App Store Review Checklist

## General
- [ ] App is fully functional (no crashes, no placeholder content)
- [ ] App requires user account? Provide demo credentials
- [ ] App uses third-party content? Provide proper attributions
- [ ] Privacy policy URL is valid and accessible
- [ ] Support URL provides adequate contact information

## Performance
- [ ] App launches in under 10 seconds
- [ ] No visible lag or jank during navigation
- [ ] App handles low memory conditions gracefully
- [ ] Battery usage is reasonable

## Security
- [ ] HTTPS used for all network calls
- [ ] Sensitive data stored securely (Keychain)
- [ ] No hardcoded credentials or API keys
- [ ] Certificate pinning implemented (if applicable)

## Privacy
- [ ] Privacy manifest (PrivacyInfo.xcprivacy) included
- [ ] Permission requests have clear usage descriptions
- [ ] User data handling described in privacy policy
- [ ] No tracking without user consent

## User Interface
- [ ] App supports all required device orientations
- [ ] App adapts to different screen sizes
- [ ] Dark mode support (if targeting iOS 13+)
- [ ] Dynamic Type support for accessibility

## Legal
- [ ] Age rating is accurate
- [ ] Content rights secured (music, images, text)
- [ ] EULA included (if applicable)
- [ ] Export compliance declaration complete
```

---

## Google Play Store Deployment

### App Signing and Build Configuration

**Keystore Generation:**

```bash
# Generate release keystore
keytool -genkey -v \
  -keystore my-release-key.keystore \
  -alias my-key-alias \
  -keyalg RSA \
  -keysize 2048 \
  -validity 10000

# Verify keystore
keytool -list -v -keystore my-release-key.keystore
```

**Gradle Configuration:**

```kotlin
// app/build.gradle.kts
android {
    signingConfigs {
        create("release") {
            storeFile = file(System.getenv("KEYSTORE_FILE") ?: "release.keystore")
            storePassword = System.getenv("KEYSTORE_PASSWORD")
            keyAlias = System.getenv("KEY_ALIAS")
            keyPassword = System.getenv("KEY_PASSWORD")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            signingConfig = signingConfigs.getByName("release")
        }
    }

    bundle {
        language {
            enableSplit = true
        }
        density {
            enableSplit = true
        }
        abi {
            enableSplit = true
        }
    }
}
```

**ProGuard Rules:**

```proguard
# app/proguard-rules.pro
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt

# Keep data classes
-keep class com.example.data.models.** { *; }

# Keep Retrofit interfaces
-keep interface com.example.data.remote.** { *; }

# Keep Compose
-keep class androidx.compose.** { *; }

# Kotlinx Serialization
-keepclassmembers class kotlinx.serialization.json.** {
    *** Companion;
}
-keepclasseswithmembers class kotlinx.serialization.json.** {
    kotlinx.serialization.KSerializer serializer(...);
}

# Keep enums
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}
```

### Play Console Configuration

**Fastlane Setup:**

```ruby
# fastlane/Fastfile
default_platform(:android)

platform :android do
  desc "Build release AAB"
  lane :build do
    gradle(
      task: "clean bundleRelease",
      properties: {
        "android.injected.signing.store.file" => ENV['KEYSTORE_FILE'],
        "android.injected.signing.store.password" => ENV['KEYSTORE_PASSWORD'],
        "android.injected.signing.key.alias" => ENV['KEY_ALIAS'],
        "android.injected.signing.key.password" => ENV['KEY_PASSWORD']
      }
    )
  end

  desc "Deploy to internal testing"
  lane :internal do
    build

    upload_to_play_store(
      track: 'internal',
      aab: 'app/build/outputs/bundle/release/app-release.aab',
      skip_upload_metadata: true,
      skip_upload_images: true,
      skip_upload_screenshots: true
    )

    slack(
      message: "New internal test build uploaded to Play Console",
      success: true
    )
  end

  desc "Deploy to beta"
  lane :beta do
    build

    upload_to_play_store(
      track: 'beta',
      aab: 'app/build/outputs/bundle/release/app-release.aab',
      rollout: '0.5' # 50% rollout
    )
  end

  desc "Deploy to production"
  lane :release do
    build

    upload_to_play_store(
      track: 'production',
      aab: 'app/build/outputs/bundle/release/app-release.aab',
      rollout: '0.1', # Start with 10%
      release_status: 'draft'
    )
  end
end
```

**Play Store Listing Metadata:**

```yaml
# metadata/android/en-US/title.txt
MyApp - Task Manager

# metadata/android/en-US/short_description.txt
Powerful cross-platform task management with real-time sync

# metadata/android/en-US/full_description.txt
MyApp is your ultimate productivity companion, designed to help you manage tasks efficiently across all your devices.

Key Features:
• User profile customization
• Real-time synchronization across devices
• Offline mode with automatic sync
• Beautiful Material You design
• Dark mode support
• Accessible and inclusive interface

Whether you're managing personal tasks or coordinating with a team, MyApp provides the tools you need to stay organized and productive.

Privacy First:
Your data is encrypted and stored securely. We never sell your information to third parties.

# metadata/android/en-US/changelogs/12.txt
Version 1.2.0 Release Notes:

- Enhanced profile editing with real-time validation
- Improved offline sync reliability
- Material You dynamic theming
- Performance optimizations
- Bug fixes

Thank you for using MyApp!
```

### Staged Rollout Strategy

```ruby
# fastlane/Fastfile
desc "Increase production rollout"
lane :increase_rollout do |options|
  current_percentage = options[:from] || 0.1
  target_percentage = options[:to] || 1.0

  upload_to_play_store(
    track: 'production',
    rollout: target_percentage.to_s,
    skip_upload_aab: true,
    skip_upload_apk: true,
    skip_upload_metadata: true,
    skip_upload_images: true,
    skip_upload_screenshots: true
  )

  slack(
    message: "Production rollout increased from #{(current_percentage * 100).to_i}% to #{(target_percentage * 100).to_i}%",
    success: true
  )
end

# Usage:
# fastlane increase_rollout from:0.1 to:0.25  # 10% → 25%
# fastlane increase_rollout from:0.25 to:0.5  # 25% → 50%
# fastlane increase_rollout from:0.5 to:1.0   # 50% → 100%
```

---

## Web Deployment

### Vercel Deployment

**Configuration:**

```json
// vercel.json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "framework": "nextjs",
  "regions": ["iad1", "sfo1"],
  "env": {
    "NEXT_PUBLIC_API_URL": "@api-url",
    "DATABASE_URL": "@database-url",
    "JWT_SECRET": "@jwt-secret"
  },
  "build": {
    "env": {
      "NEXT_TELEMETRY_DISABLED": "1"
    }
  },
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        },
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        },
        {
          "key": "X-XSS-Protection",
          "value": "1; mode=block"
        },
        {
          "key": "Referrer-Policy",
          "value": "strict-origin-when-cross-origin"
        },
        {
          "key": "Permissions-Policy",
          "value": "camera=(), microphone=(), geolocation=()"
        }
      ]
    }
  ],
  "redirects": [
    {
      "source": "/home",
      "destination": "/",
      "permanent": true
    }
  ],
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://api.example.com/:path*"
    }
  ]
}
```

**GitHub Actions for Vercel:**

```yaml
# .github/workflows/deploy.yml
name: Deploy to Vercel

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
  VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install Vercel CLI
        run: npm install -g vercel@latest

      - name: Pull Vercel Environment
        run: vercel pull --yes --environment=production --token=${{ secrets.VERCEL_TOKEN }}

      - name: Build Project
        run: vercel build --prod --token=${{ secrets.VERCEL_TOKEN }}

      - name: Deploy to Production
        if: github.ref == 'refs/heads/main'
        run: vercel deploy --prebuilt --prod --token=${{ secrets.VERCEL_TOKEN }}

      - name: Deploy to Preview
        if: github.ref != 'refs/heads/main'
        run: vercel deploy --prebuilt --token=${{ secrets.VERCEL_TOKEN }}
```

### Self-Hosted Docker Deployment

**Dockerfile:**

```dockerfile
# Dockerfile
FROM node:20-alpine AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

CMD ["node", "server.js"]
```

**Docker Compose:**

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=https://api.example.com
      - DATABASE_URL=postgresql://user:pass@db:5432/myapp
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=myapp
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - web
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### CDN Configuration (Cloudflare)

**Terraform Configuration:**

```hcl
# cloudflare.tf
terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

resource "cloudflare_zone" "example" {
  account_id = var.cloudflare_account_id
  zone       = "example.com"
}

resource "cloudflare_record" "root" {
  zone_id = cloudflare_zone.example.id
  name    = "@"
  value   = "example.com"
  type    = "CNAME"
  proxied = true
}

resource "cloudflare_record" "www" {
  zone_id = cloudflare_zone.example.id
  name    = "www"
  value   = "example.com"
  type    = "CNAME"
  proxied = true
}

resource "cloudflare_page_rule" "cache_everything" {
  zone_id = cloudflare_zone.example.id
  target  = "example.com/*"
  priority = 1

  actions {
    cache_level = "cache_everything"
    edge_cache_ttl = 7200
    browser_cache_ttl = 3600
  }
}

resource "cloudflare_page_rule" "force_https" {
  zone_id = cloudflare_zone.example.id
  target  = "http://*example.com/*"
  priority = 2

  actions {
    always_use_https = true
  }
}
```

---

## Desktop Distribution

### Electron/Tauri Code Signing

**macOS Code Signing:**

```json
// package.json
{
  "build": {
    "appId": "com.example.myapp",
    "mac": {
      "category": "public.app-category.productivity",
      "hardenedRuntime": true,
      "gatekeeperAssess": false,
      "entitlements": "build/entitlements.mac.plist",
      "entitlementsInherit": "build/entitlements.mac.plist",
      "provisioningProfile": "build/MyApp.provisionprofile",
      "target": [
        {
          "target": "dmg",
          "arch": ["x64", "arm64"]
        },
        {
          "target": "zip",
          "arch": ["x64", "arm64"]
        }
      ]
    },
    "dmg": {
      "contents": [
        {
          "x": 130,
          "y": 220
        },
        {
          "x": 410,
          "y": 220,
          "type": "link",
          "path": "/Applications"
        }
      ],
      "sign": true
    },
    "afterSign": "scripts/notarize.js"
  }
}
```

**Notarization Script:**

```javascript
// scripts/notarize.js
const { notarize } = require('@electron/notarize');

exports.default = async function notarizing(context) {
  const { electronPlatformName, appOutDir } = context;

  if (electronPlatformName !== 'darwin') {
    return;
  }

  const appName = context.packager.appInfo.productFilename;

  return await notarize({
    appBundleId: 'com.example.myapp',
    appPath: `${appOutDir}/${appName}.app`,
    appleId: process.env.APPLE_ID,
    appleIdPassword: process.env.APPLE_ID_PASSWORD,
    teamId: process.env.APPLE_TEAM_ID
  });
};
```

**Windows Code Signing:**

```json
{
  "build": {
    "win": {
      "target": [
        {
          "target": "nsis",
          "arch": ["x64", "arm64"]
        }
      ],
      "certificateFile": "./certs/cert.pfx",
      "certificatePassword": "${env.WINDOWS_CERT_PASSWORD}",
      "publisherName": "Example Inc.",
      "signDlls": true
    },
    "nsis": {
      "oneClick": false,
      "allowToChangeInstallationDirectory": true,
      "createDesktopShortcut": true,
      "createStartMenuShortcut": true,
      "shortcutName": "MyApp"
    }
  }
}
```

### Auto-Update Configuration

**Electron Builder Auto-Update:**

```typescript
// src/main/autoUpdater.ts
import { autoUpdater } from 'electron-updater';
import { app, BrowserWindow } from 'electron';
import log from 'electron-log';

export function setupAutoUpdater(mainWindow: BrowserWindow) {
  autoUpdater.logger = log;
  autoUpdater.logger.transports.file.level = 'info';

  autoUpdater.on('checking-for-update', () => {
    log.info('Checking for updates...');
    mainWindow.webContents.send('update-checking');
  });

  autoUpdater.on('update-available', (info) => {
    log.info('Update available:', info);
    mainWindow.webContents.send('update-available', info);
  });

  autoUpdater.on('update-not-available', (info) => {
    log.info('Update not available:', info);
    mainWindow.webContents.send('update-not-available', info);
  });

  autoUpdater.on('error', (err) => {
    log.error('Error in auto-updater:', err);
    mainWindow.webContents.send('update-error', err);
  });

  autoUpdater.on('download-progress', (progressObj) => {
    mainWindow.webContents.send('download-progress', progressObj);
  });

  autoUpdater.on('update-downloaded', (info) => {
    log.info('Update downloaded:', info);
    mainWindow.webContents.send('update-downloaded', info);
  });

  // Check for updates on app startup
  app.whenReady().then(() => {
    setTimeout(() => autoUpdater.checkForUpdatesAndNotify(), 3000);
  });

  // Check for updates every 4 hours
  setInterval(() => {
    autoUpdater.checkForUpdatesAndNotify();
  }, 4 * 60 * 60 * 1000);
}
```

**Update Server Configuration:**

```yaml
# releases/latest.yml
version: 1.2.0
files:
  - url: MyApp-1.2.0-mac-x64.zip
    sha512: abc123...
    size: 89456123
  - url: MyApp-1.2.0-mac-arm64.zip
    sha512: def456...
    size: 92145678
path: MyApp-1.2.0-mac-x64.zip
sha512: abc123...
releaseDate: '2025-01-15T10:00:00.000Z'
```

---

## CI/CD Pipelines

### GitHub Actions Multi-Platform Pipeline

```yaml
# .github/workflows/ci-cd.yml
name: Multi-Platform CI/CD

on:
  push:
    branches: [main, develop]
    tags: ['v*']
  pull_request:
    branches: [main]

jobs:
  web:
    name: Web Build & Deploy
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test

      - name: Build
        run: npm run build
        env:
          NEXT_PUBLIC_API_URL: ${{ secrets.API_URL }}

      - name: Deploy to Vercel
        if: github.ref == 'refs/heads/main'
        run: vercel deploy --prod --token=${{ secrets.VERCEL_TOKEN }}

  ios:
    name: iOS Build & Deploy
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4

      - name: Setup Xcode
        uses: maxim-lobanov/setup-xcode@v1
        with:
          xcode-version: '15.2'

      - name: Install dependencies
        run: |
          gem install bundler
          bundle install

      - name: Setup certificates
        env:
          MATCH_PASSWORD: ${{ secrets.MATCH_PASSWORD }}
          MATCH_GIT_BASIC_AUTHORIZATION: ${{ secrets.MATCH_GIT_TOKEN }}
        run: bundle exec fastlane ios sync_certificates

      - name: Run tests
        run: bundle exec fastlane ios test

      - name: Build and upload to TestFlight
        if: startsWith(github.ref, 'refs/tags/v')
        env:
          APP_STORE_CONNECT_API_KEY: ${{ secrets.APP_STORE_CONNECT_API_KEY }}
        run: bundle exec fastlane ios beta

  android:
    name: Android Build & Deploy
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Java
        uses: actions/setup-java@v4
        with:
          distribution: 'temurin'
          java-version: '17'

      - name: Setup Android SDK
        uses: android-actions/setup-android@v3

      - name: Install dependencies
        run: |
          gem install bundler
          bundle install

      - name: Run tests
        run: ./gradlew test

      - name: Build release AAB
        env:
          KEYSTORE_FILE: ${{ secrets.KEYSTORE_FILE }}
          KEYSTORE_PASSWORD: ${{ secrets.KEYSTORE_PASSWORD }}
          KEY_ALIAS: ${{ secrets.KEY_ALIAS }}
          KEY_PASSWORD: ${{ secrets.KEY_PASSWORD }}
        run: bundle exec fastlane android build

      - name: Upload to Play Console
        if: startsWith(github.ref, 'refs/tags/v')
        env:
          PLAY_STORE_JSON_KEY: ${{ secrets.PLAY_STORE_JSON_KEY }}
        run: bundle exec fastlane android internal

  desktop:
    name: Desktop Build
    strategy:
      matrix:
        os: [macos-latest, ubuntu-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Build desktop app
        run: npm run build:desktop
        env:
          APPLE_ID: ${{ secrets.APPLE_ID }}
          APPLE_ID_PASSWORD: ${{ secrets.APPLE_ID_PASSWORD }}
          WINDOWS_CERT_PASSWORD: ${{ secrets.WINDOWS_CERT_PASSWORD }}

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: desktop-${{ matrix.os }}
          path: dist/
```

---

This deployment and distribution guide provides comprehensive strategies for deploying multi-platform applications to App Store, Play Store, web platforms, and desktop distribution channels with automated CI/CD pipelines.
