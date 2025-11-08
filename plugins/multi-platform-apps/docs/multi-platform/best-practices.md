# Best Practices Guide

Comprehensive best practices for building performant, secure, and maintainable multi-platform applications covering bundle optimization, startup performance, offline-first patterns, security, and performance budgets.

## Table of Contents

1. [Bundle Size Optimization](#bundle-size-optimization)
2. [Startup Time Optimization](#startup-time-optimization)
3. [Offline-First Patterns](#offline-first-patterns)
4. [Security Best Practices](#security-best-practices)
5. [Performance Budgets](#performance-budgets)

---

## Bundle Size Optimization

### Web Bundle Optimization

**Next.js Bundle Analysis:**

```javascript
// next.config.js
const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true'
});

module.exports = withBundleAnalyzer({
  webpack: (config, { webpack }) => {
    // Enable tree shaking
    config.optimization.usedExports = true;

    // Analyze bundle
    config.plugins.push(
      new webpack.optimize.LimitChunkCountPlugin({
        maxChunks: 1
      })
    );

    return config;
  },

  // Enable SWC minification
  swcMinify: true,

  // Compression
  compress: true,

  // Remove source maps in production
  productionBrowserSourceMaps: false,

  // Image optimization
  images: {
    formats: ['image/avif', 'image/webp'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384]
  }
});
```

**Dynamic Imports and Code Splitting:**

```typescript
// components/HeavyComponent.tsx - Bad
import HeavyLibrary from 'heavy-library';

export function Component() {
  return <HeavyLibrary />;
}

// components/HeavyComponent.tsx - Good
import dynamic from 'next/dynamic';

const HeavyLibrary = dynamic(() => import('heavy-library'), {
  loading: () => <div>Loading...</div>,
  ssr: false // Disable SSR for client-only components
});

export function Component() {
  return <HeavyLibrary />;
}
```

**Route-Based Code Splitting:**

```typescript
// app/profile/page.tsx
import dynamic from 'next/dynamic';

const ProfileEditor = dynamic(() => import('@/components/features/ProfileEditor'), {
  loading: () => <ProfileSkeleton />
});

const ProfileStats = dynamic(() => import('@/components/features/ProfileStats'));
const ProfileFeed = dynamic(() => import('@/components/features/ProfileFeed'));

export default function ProfilePage() {
  return (
    <div>
      <ProfileEditor />
      <ProfileStats />
      <ProfileFeed />
    </div>
  );
}
```

**Tree Shaking Optimization:**

```typescript
// Bad - imports entire library
import _ from 'lodash';
const result = _.debounce(fn, 300);

// Good - imports only needed function
import debounce from 'lodash/debounce';
const result = debounce(fn, 300);

// Better - use modern alternatives
import { debounce } from 'es-toolkit';
const result = debounce(fn, 300);
```

**Font Optimization:**

```typescript
// app/layout.tsx
import { Inter } from 'next/font/google';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
  preload: true,
  fallback: ['system-ui', 'arial']
});

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
```

**Image Optimization:**

```typescript
// components/OptimizedImage.tsx
import Image from 'next/image';

export function OptimizedImage({ src, alt }: { src: string; alt: string }) {
  return (
    <Image
      src={src}
      alt={alt}
      width={800}
      height={600}
      quality={85}
      placeholder="blur"
      blurDataURL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2w..."
      priority={false} // Only true for LCP images
      sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
    />
  );
}
```

### iOS Bundle Optimization

**App Thinning and On-Demand Resources:**

```swift
// Build Settings
/*
Enable Bitcode: Yes (for App Store optimization)
Enable App Slicing: Yes
Strip Debug Symbols: Yes (Release)
Strip Swift Symbols: Yes (Release)
Make Strings Read-Only: Yes
Dead Code Stripping: Yes
Asset Catalog Compiler - Optimization: space
*/

// On-Demand Resources for large assets
// Assets.xcassets -> Tags: level1, level2, level3

// ContentManager.swift
import Foundation

class ContentManager {
    func downloadLevel(_ level: Int) async throws {
        let resourceTag = "level\(level)"

        let request = NSBundleResourceRequest(tags: [resourceTag])
        request.loadingPriority = NSBundleResourceRequestLoadingPriorityUrgent

        try await request.beginAccessingResources()

        // Use resources
        // ...

        // Release when done
        request.endAccessingResources()
    }
}
```

**SwiftUI View Hierarchy Optimization:**

```swift
// Bad - Creates unnecessary views
struct ProfileView: View {
    var body: some View {
        VStack {
            VStack {
                VStack {
                    Text("Name")
                }
            }
        }
    }
}

// Good - Flattened hierarchy
struct ProfileView: View {
    var body: some View {
        VStack {
            Text("Name")
        }
    }
}

// Better - Use @ViewBuilder for conditional content
struct ProfileView: View {
    let isEditing: Bool

    var body: some View {
        VStack {
            profileContent
        }
    }

    @ViewBuilder
    private var profileContent: some View {
        if isEditing {
            ProfileEditor()
        } else {
            ProfileDisplay()
        }
    }
}
```

### Android APK/AAB Optimization

**ProGuard Optimization:**

```proguard
# app/proguard-rules.pro
-optimizationpasses 5
-dontusemixedcaseclassnames
-dontskipnonpubliclibraryclasses
-dontpreverify
-verbose

# Remove logging
-assumenosideeffects class android.util.Log {
    public static *** d(...);
    public static *** v(...);
    public static *** i(...);
}

# Optimize for size
-repackageclasses ''
-allowaccessmodification
-optimizations !code/simplification/arithmetic,!code/simplification/cast,!field/*,!class/merging/*
```

**R8 Full Mode:**

```gradle
// gradle.properties
android.enableR8.fullMode=true
android.enableR8=true
android.enableDexingArtifactTransform=true
```

**Resource Shrinking:**

```kotlin
// build.gradle.kts
android {
    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true

            // Remove unused resources
            resourceConfigurations += setOf("en", "xxhdpi")
        }
    }

    packagingOptions {
        resources {
            // Exclude duplicate files
            excludes += setOf(
                "META-INF/LICENSE",
                "META-INF/NOTICE",
                "META-INF/*.kotlin_module"
            )
        }
    }
}
```

**Vector Drawables:**

```xml
<!-- res/drawable/ic_profile.xml - Instead of PNG -->
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="24dp"
    android:height="24dp"
    android:viewportWidth="24"
    android:viewportHeight="24">
    <path
        android:fillColor="@android:color/white"
        android:pathData="M12,12c2.21,0 4,-1.79 4,-4s-1.79,-4 -4,-4 -4,1.79 -4,4 1.79,4 4,4zM12,14c-2.67,0 -8,1.34 -8,4v2h16v-2c0,-2.66 -5.33,-4 -8,-4z"/>
</vector>
```

---

## Startup Time Optimization

### Web First Paint Optimization

**Critical CSS Inlining:**

```typescript
// app/layout.tsx
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html>
      <head>
        <style dangerouslySetInnerHTML={{
          __html: `
            /* Critical CSS - Above the fold */
            body { margin: 0; font-family: system-ui; }
            .header { height: 60px; background: #fff; }
            .skeleton { background: #f0f0f0; animation: pulse 2s infinite; }
            @keyframes pulse {
              0%, 100% { opacity: 1; }
              50% { opacity: 0.5; }
            }
          `
        }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
```

**Lazy Loading Images:**

```typescript
// components/LazyImage.tsx
'use client';

import { useState, useEffect, useRef } from 'react';

export function LazyImage({ src, alt, ...props }: React.ImgHTMLAttributes<HTMLImageElement>) {
  const [isLoaded, setIsLoaded] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (!imgRef.current) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsLoaded(true);
          observer.disconnect();
        }
      },
      { rootMargin: '50px' }
    );

    observer.observe(imgRef.current);

    return () => observer.disconnect();
  }, []);

  return (
    <img
      ref={imgRef}
      src={isLoaded ? src : undefined}
      alt={alt}
      loading="lazy"
      {...props}
    />
  );
}
```

### iOS Launch Time Optimization

**Optimize App Delegate:**

```swift
// AppDelegate.swift
import UIKit

@main
class AppDelegate: UIResponder, UIApplicationDelegate {

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        // Defer non-critical initialization
        DispatchQueue.global(qos: .utility).async {
            self.setupAnalytics()
            self.setupCrashReporting()
            self.preloadCaches()
        }

        // Only critical initialization here
        setupAppearance()

        return true
    }

    private func setupAppearance() {
        // Fast, synchronous appearance setup
        UINavigationBar.appearance().tintColor = .brandPrimary
    }

    private func setupAnalytics() {
        // Deferred analytics initialization
        Analytics.configure()
    }

    private func setupCrashReporting() {
        // Deferred crash reporting
        CrashReporter.configure()
    }

    private func preloadCaches() {
        // Warm up caches in background
        CacheService.shared.warmup()
    }
}
```

**Lazy Property Initialization:**

```swift
// Services/UserService.swift
class UserService {
    // Bad - initialized immediately
    let database = Database()
    let apiClient = APIClient()

    // Good - lazy initialization
    lazy var database: Database = {
        let config = DatabaseConfig()
        return Database(config: config)
    }()

    lazy var apiClient: APIClient = {
        let config = APIConfig()
        return APIClient(config: config)
    }()
}
```

### Android Startup Optimization

**App Startup Library:**

```kotlin
// app/build.gradle.kts
dependencies {
    implementation("androidx.startup:startup-runtime:1.1.1")
}

// AppInitializer.kt
package com.example.initializers

import android.content.Context
import androidx.startup.Initializer
import com.example.analytics.Analytics
import com.example.logging.Logger

class AnalyticsInitializer : Initializer<Analytics> {
    override fun create(context: Context): Analytics {
        return Analytics.getInstance(context).apply {
            initialize()
        }
    }

    override fun dependencies(): List<Class<out Initializer<*>>> {
        return listOf(LoggerInitializer::class.java)
    }
}

class LoggerInitializer : Initializer<Logger> {
    override fun create(context: Context): Logger {
        return Logger.getInstance(context).apply {
            configure()
        }
    }

    override fun dependencies(): List<Class<out Initializer<*>>> {
        return emptyList()
    }
}

// AndroidManifest.xml
<provider
    android:name="androidx.startup.InitializationProvider"
    android:authorities="${applicationId}.androidx-startup"
    android:exported="false"
    tools:node="merge">
    <meta-data
        android:name="com.example.initializers.AnalyticsInitializer"
        android:value="androidx.startup" />
</provider>
```

**Lazy Injection with Hilt:**

```kotlin
// di/AppModule.kt
@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideDatabase(
        @ApplicationContext context: Context
    ): Lazy<AppDatabase> = lazy {
        Room.databaseBuilder(
            context,
            AppDatabase::class.java,
            "app-database"
        ).build()
    }

    @Provides
    @Singleton
    fun provideApiService(): Lazy<ApiService> = lazy {
        Retrofit.Builder()
            .baseUrl("https://api.example.com")
            .addConverterFactory(KotlinxSerializationConverterFactory.create(Json))
            .build()
            .create(ApiService::class.java)
    }
}

// Usage
@Inject
lateinit var database: Lazy<AppDatabase>

fun loadData() {
    // Database only initialized when first accessed
    val users = database.value.userDao().getAll()
}
```

---

## Offline-First Patterns

### Service Worker Strategies

**Network-First with Cache Fallback:**

```typescript
// public/sw.js
self.addEventListener('fetch', (event) => {
  if (event.request.url.includes('/api/')) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          // Clone and cache the response
          const responseToCache = response.clone();
          caches.open('api-cache').then((cache) => {
            cache.put(event.request, responseToCache);
          });
          return response;
        })
        .catch(() => {
          // Network failed, try cache
          return caches.match(event.request).then((response) => {
            if (response) {
              return response;
            }
            // Return offline page
            return new Response(
              JSON.stringify({ error: 'Offline' }),
              {
                status: 503,
                headers: { 'Content-Type': 'application/json' }
              }
            );
          });
        })
    );
  }
});
```

**Background Sync:**

```typescript
// lib/sync/backgroundSync.ts
export async function registerBackgroundSync() {
  if ('serviceWorker' in navigator && 'SyncManager' in window) {
    const registration = await navigator.serviceWorker.ready;
    await registration.sync.register('sync-data');
  }
}

// public/sw.js
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-data') {
    event.waitUntil(syncPendingRequests());
  }
});

async function syncPendingRequests() {
  const pendingRequests = await getPendingRequests();

  for (const request of pendingRequests) {
    try {
      const response = await fetch(request.url, {
        method: request.method,
        body: request.body,
        headers: request.headers
      });

      if (response.ok) {
        await removePendingRequest(request.id);
      }
    } catch (error) {
      console.error('Sync failed:', error);
    }
  }
}
```

### iOS Offline Data Management

**Core Data with NSPersistentCloudKitContainer:**

```swift
// Services/PersistenceController.swift
import CoreData
import CloudKit

class PersistenceController {
    static let shared = PersistenceController()

    let container: NSPersistentCloudKitContainer

    init() {
        container = NSPersistentCloudKitContainer(name: "MyApp")

        // Configure CloudKit sync
        guard let description = container.persistentStoreDescriptions.first else {
            fatalError("No persistent store descriptions found")
        }

        description.setOption(true as NSNumber, forKey: NSPersistentHistoryTrackingKey)
        description.setOption(true as NSNumber, forKey: NSPersistentStoreRemoteChangeNotificationPostOptionKey)

        // Enable background updates
        description.cloudKitContainerOptions = NSPersistentCloudKitContainerOptions(
            containerIdentifier: "iCloud.com.example.myapp"
        )

        container.loadPersistentStores { description, error in
            if let error = error {
                fatalError("Core Data failed to load: \(error.localizedDescription)")
            }
        }

        container.viewContext.automaticallyMergesChangesFromParent = true
        container.viewContext.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy

        // Observe remote changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(processRemoteStoreChange),
            name: .NSPersistentStoreRemoteChange,
            object: container.persistentStoreCoordinator
        )
    }

    @objc
    private func processRemoteStoreChange(_ notification: Notification) {
        // Process CloudKit changes
        print("Remote store changed")
    }

    func save() {
        let context = container.viewContext
        if context.hasChanges {
            do {
                try context.save()
            } catch {
                let nsError = error as NSError
                fatalError("Unresolved error \(nsError), \(nsError.userInfo)")
            }
        }
    }
}
```

### Android Offline Sync with WorkManager

```kotlin
// sync/SyncWorker.kt
package com.example.sync

import android.content.Context
import androidx.work.*
import com.example.data.local.PendingRequestDao
import com.example.data.remote.ApiService
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.TimeUnit
import javax.inject.Inject

class SyncWorker @Inject constructor(
    @ApplicationContext context: Context,
    params: WorkerParameters,
    private val pendingRequestDao: PendingRequestDao,
    private val apiService: ApiService
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        try {
            val pendingRequests = pendingRequestDao.getAll()

            for (request in pendingRequests) {
                try {
                    when (request.method) {
                        "POST" -> apiService.post(request.url, request.body)
                        "PUT" -> apiService.put(request.url, request.body)
                        "PATCH" -> apiService.patch(request.url, request.body)
                        "DELETE" -> apiService.delete(request.url)
                    }

                    pendingRequestDao.delete(request)
                } catch (e: Exception) {
                    // Request failed, keep in queue
                    continue
                }
            }

            Result.success()
        } catch (e: Exception) {
            Result.retry()
        }
    }

    companion object {
        fun schedulePeriodic(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .setRequiresBatteryNotLow(true)
                .build()

            val request = PeriodicWorkRequestBuilder<SyncWorker>(
                15, TimeUnit.MINUTES
            )
                .setConstraints(constraints)
                .setBackoffCriteria(
                    BackoffPolicy.EXPONENTIAL,
                    WorkRequest.MIN_BACKOFF_MILLIS,
                    TimeUnit.MILLISECONDS
                )
                .build()

            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                "sync-worker",
                ExistingPeriodicWorkPolicy.KEEP,
                request
            )
        }
    }
}
```

---

## Security Best Practices

### Secure Data Storage

**Web - Encrypted Local Storage:**

```typescript
// lib/security/secureStorage.ts
import CryptoJS from 'crypto-js';

export class SecureStorage {
  private static getEncryptionKey(): string {
    // Generate from user session or derive from password
    return sessionStorage.getItem('encryption-key') || '';
  }

  static set(key: string, value: any): void {
    const encryptionKey = this.getEncryptionKey();
    const encrypted = CryptoJS.AES.encrypt(
      JSON.stringify(value),
      encryptionKey
    ).toString();

    localStorage.setItem(key, encrypted);
  }

  static get<T>(key: string): T | null {
    const encrypted = localStorage.getItem(key);
    if (!encrypted) return null;

    try {
      const encryptionKey = this.getEncryptionKey();
      const decrypted = CryptoJS.AES.decrypt(encrypted, encryptionKey);
      return JSON.parse(decrypted.toString(CryptoJS.enc.Utf8));
    } catch {
      return null;
    }
  }

  static remove(key: string): void {
    localStorage.removeItem(key);
  }

  static clear(): void {
    localStorage.clear();
  }
}
```

**iOS - Keychain Storage:**

```swift
// Services/KeychainService.swift
import Security
import Foundation

class KeychainService {
    static let shared = KeychainService()

    enum KeychainError: Error {
        case duplicateItem
        case unknown(OSStatus)
    }

    func save(key: String, data: Data) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock
        ]

        let status = SecItemAdd(query as CFDictionary, nil)

        guard status != errSecDuplicateItem else {
            throw KeychainError.duplicateItem
        }

        guard status == errSecSuccess else {
            throw KeychainError.unknown(status)
        }
    }

    func load(key: String) throws -> Data {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess else {
            throw KeychainError.unknown(status)
        }

        guard let data = result as? Data else {
            throw KeychainError.unknown(errSecInvalidData)
        }

        return data
    }

    func delete(key: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]

        let status = SecItemDelete(query as CFDictionary)

        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.unknown(status)
        }
    }

    // Convenience methods
    func saveString(key: String, value: String) throws {
        guard let data = value.data(using: .utf8) else { return }
        try save(key: key, data: data)
    }

    func loadString(key: String) throws -> String {
        let data = try load(key: key)
        guard let string = String(data: data, encoding: .utf8) else {
            throw KeychainError.unknown(errSecInvalidData)
        }
        return string
    }
}
```

**Android - EncryptedSharedPreferences:**

```kotlin
// security/SecureStorage.kt
package com.example.security

import android.content.Context
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class SecureStorage @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private val masterKey = MasterKey.Builder(context)
        .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
        .build()

    private val sharedPreferences = EncryptedSharedPreferences.create(
        context,
        "secure_prefs",
        masterKey,
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
    )

    fun saveString(key: String, value: String) {
        sharedPreferences.edit().putString(key, value).apply()
    }

    fun getString(key: String, defaultValue: String? = null): String? {
        return sharedPreferences.getString(key, defaultValue)
    }

    fun saveBoolean(key: String, value: Boolean) {
        sharedPreferences.edit().putBoolean(key, value).apply()
    }

    fun getBoolean(key: String, defaultValue: Boolean = false): Boolean {
        return sharedPreferences.getBoolean(key, defaultValue)
    }

    fun remove(key: String) {
        sharedPreferences.edit().remove(key).apply()
    }

    fun clear() {
        sharedPreferences.edit().clear().apply()
    }
}
```

### Certificate Pinning

**iOS:**

```swift
// Services/NetworkSecurityManager.swift
import Foundation

class NetworkSecurityManager: NSObject, URLSessionDelegate {
    private let pinnedCertificates: [SecCertificate]

    init(pinnedCertificateNames: [String]) {
        var certificates: [SecCertificate] = []

        for name in pinnedCertificateNames {
            if let path = Bundle.main.path(forResource: name, ofType: "cer"),
               let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
               let certificate = SecCertificateCreateWithData(nil, data as CFData) {
                certificates.append(certificate)
            }
        }

        self.pinnedCertificates = certificates
        super.init()
    }

    func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        guard let serverTrust = challenge.protectionSpace.serverTrust else {
            completionHandler(.cancelAuthenticationChallenge, nil)
            return
        }

        let policies = [SecPolicyCreateSSL(true, challenge.protectionSpace.host as CFString)]
        SecTrustSetPolicies(serverTrust, policies as CFArray)

        guard let serverCertificate = SecTrustGetCertificateAtIndex(serverTrust, 0) else {
            completionHandler(.cancelAuthenticationChallenge, nil)
            return
        }

        for pinnedCertificate in pinnedCertificates {
            if serverCertificate == pinnedCertificate {
                completionHandler(.useCredential, URLCredential(trust: serverTrust))
                return
            }
        }

        completionHandler(.cancelAuthenticationChallenge, nil)
    }
}
```

**Android:**

```kotlin
// security/CertificatePinner.kt
package com.example.security

import okhttp3.CertificatePinner
import okhttp3.OkHttpClient

object NetworkConfig {
    fun createSecureClient(): OkHttpClient {
        val certificatePinner = CertificatePinner.Builder()
            .add(
                "api.example.com",
                "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
            )
            .add(
                "api.example.com",
                "sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=" // Backup pin
            )
            .build()

        return OkHttpClient.Builder()
            .certificatePinner(certificatePinner)
            .build()
    }
}
```

---

## Performance Budgets

### Web Performance Budgets

```json
// budget.json
{
  "budgets": [
    {
      "path": "/*",
      "timings": [
        {
          "metric": "first-contentful-paint",
          "budget": 2000
        },
        {
          "metric": "largest-contentful-paint",
          "budget": 2500
        },
        {
          "metric": "cumulative-layout-shift",
          "budget": 0.1
        },
        {
          "metric": "total-blocking-time",
          "budget": 300
        },
        {
          "metric": "time-to-interactive",
          "budget": 3500
        }
      ],
      "resourceSizes": [
        {
          "resourceType": "script",
          "budget": 300
        },
        {
          "resourceType": "stylesheet",
          "budget": 50
        },
        {
          "resourceType": "image",
          "budget": 500
        },
        {
          "resourceType": "font",
          "budget": 100
        },
        {
          "resourceType": "total",
          "budget": 1000
        }
      ],
      "resourceCounts": [
        {
          "resourceType": "script",
          "budget": 15
        },
        {
          "resourceType": "stylesheet",
          "budget": 5
        },
        {
          "resourceType": "third-party",
          "budget": 10
        }
      ]
    }
  ]
}
```

### Mobile Performance Budgets

**iOS Performance Metrics:**

```swift
// Tests/PerformanceBudgets.swift
import XCTest

final class PerformanceBudgets: XCTestCase {
    let budgets = PerformanceBudget(
        appLaunchTime: 1.0,        // <1 second cold start
        screenTransition: 0.3,     // <300ms screen transitions
        listScrolling: 60.0,       // 60fps scrolling
        memoryUsage: 100.0,        // <100MB memory
        batteryDrain: 2.0          // <2% per hour
    )

    func testAppLaunchPerformance() throws {
        measure(metrics: [XCTClockMetric()]) {
            XCUIApplication().launch()
        }

        // Verify against budget
        let options = XCTMeasureOptions()
        let results = try XCTContext.runActivity(named: "Launch Time") { _ in
            return measure(options: options)
        }

        XCTAssertLessThan(results.average, budgets.appLaunchTime)
    }

    func testMemoryBudget() throws {
        let app = XCUIApplication()
        app.launch()

        measure(metrics: [XCTMemoryMetric()]) {
            // Navigate through app
            app.buttons["profile-tab"].tap()
            app.buttons["feed-tab"].tap()
            app.buttons["settings-tab"].tap()
        }
    }
}
```

**Android Performance Metrics:**

```kotlin
// benchmark/PerformanceBudgets.kt
package com.example.benchmark

object PerformanceBudgets {
    // Startup
    const val COLD_START_MS = 1000L
    const val WARM_START_MS = 500L

    // UI
    const val FRAME_TIME_MS = 16L  // 60fps
    const val JANK_THRESHOLD = 0.05 // <5% janky frames

    // Memory
    const val MEMORY_USAGE_MB = 100L

    // Network
    const val API_RESPONSE_MS = 500L

    // APK Size
    const val APK_SIZE_MB = 20L
}

@RunWith(AndroidJUnit4::class)
class StartupBenchmark {
    @get:Rule
    val benchmarkRule = MacrobenchmarkRule()

    @Test
    fun startup() = benchmarkRule.measureRepeated(
        packageName = "com.example.myapp",
        metrics = listOf(StartupTimingMetric()),
        iterations = 5,
        startupMode = StartupMode.COLD
    ) {
        pressHome()
        startActivityAndWait()

        // Verify against budget
        val metrics = getMetrics()
        val startupTime = metrics["startup_time_ms"] as Long
        assertTrue(
            "Startup time $startupTime exceeds budget ${PerformanceBudgets.COLD_START_MS}",
            startupTime < PerformanceBudgets.COLD_START_MS
        )
    }
}
```

---

This best practices guide provides comprehensive patterns for optimizing bundle size, startup performance, implementing offline-first architecture, securing applications, and maintaining performance budgets across web, iOS, and Android platforms.
