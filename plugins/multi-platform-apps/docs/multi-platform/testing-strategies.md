# Testing Strategies Guide

Comprehensive testing approaches for multi-platform applications including feature parity validation, automated testing frameworks, performance benchmarking, and end-to-end testing.

## Table of Contents

1. [Feature Parity Matrix](#feature-parity-matrix)
2. [Platform Testing Frameworks](#platform-testing-frameworks)
3. [Performance Benchmarking](#performance-benchmarking)
4. [End-to-End Testing](#end-to-end-testing)
5. [Visual Regression Testing](#visual-regression-testing)

---

## Feature Parity Matrix

### Parity Validation Checklist

**Template for Feature Validation:**

```markdown
# Feature: User Profile Update

## Functional Parity

| Requirement | Web | iOS | Android | Desktop | Status |
|------------|-----|-----|---------|---------|--------|
| Display name validation | ✅ | ✅ | ✅ | ✅ | Pass |
| Bio character limit (500) | ✅ | ✅ | ✅ | ✅ | Pass |
| Avatar upload | ✅ | ✅ | ✅ | ⚠️ | Partial |
| Offline support | ✅ | ✅ | ✅ | ✅ | Pass |
| Error handling | ✅ | ✅ | ✅ | ✅ | Pass |
| Success feedback | ✅ | ✅ | ✅ | ✅ | Pass |

## UI Consistency

| Element | Web | iOS | Android | Desktop | Variance |
|---------|-----|-----|---------|---------|----------|
| Form layout | Standard | Native | Material 3 | Standard | <5% |
| Button styling | Primary | iOS style | FAB | Primary | <5% |
| Input fields | Outlined | iOS default | Outlined | Outlined | <5% |
| Character counter | Bottom right | Bottom right | Bottom right | Bottom right | 0% |
| Error messages | Below field | Below field | Below field | Below field | 0% |

## Performance

| Metric | Target | Web | iOS | Android | Desktop |
|--------|--------|-----|-----|---------|---------|
| Load time | <1s | 0.8s | 0.6s | 0.7s | 0.7s |
| Save time | <2s | 1.5s | 1.2s | 1.3s | 1.4s |
| UI responsiveness | 60fps | 60fps | 60fps | 58fps | 60fps |
| Memory usage | <50MB | 45MB | 42MB | 48MB | 46MB |

## Accessibility

| Requirement | Web | iOS | Android | Desktop | WCAG Level |
|------------|-----|-----|---------|---------|------------|
| Screen reader support | ✅ | ✅ | ✅ | ✅ | AA |
| Keyboard navigation | ✅ | ✅ | N/A | ✅ | AA |
| Color contrast | 4.8:1 | 4.8:1 | 4.8:1 | 4.8:1 | AA |
| Touch targets | 44px | 44px | 48dp | 44px | AA |
| Focus indicators | ✅ | ✅ | ✅ | ✅ | AA |

## Edge Cases

| Scenario | Web | iOS | Android | Desktop |
|----------|-----|-----|---------|---------|
| Network timeout | ✅ Retry | ✅ Retry | ✅ Retry | ✅ Retry |
| Offline mode | ✅ Queue | ✅ Queue | ✅ Queue | ✅ Queue |
| Concurrent edits | ✅ Conflict | ✅ Conflict | ✅ Conflict | ✅ Conflict |
| Empty inputs | ✅ Clear | ✅ Clear | ✅ Clear | ✅ Clear |
| Max length | ✅ Prevent | ✅ Prevent | ✅ Prevent | ✅ Prevent |
```

### Automated Parity Testing

**Cross-Platform Test Specification:**

```typescript
// tests/parity/profileUpdate.spec.ts
import { test, expect } from '@playwright/test';
import { Maestro } from 'maestro';
import { Detox } from 'detox';

interface PlatformTestSuite {
  web: () => Promise<TestResult>;
  ios: () => Promise<TestResult>;
  android: () => Promise<TestResult>;
  desktop: () => Promise<TestResult>;
}

interface TestResult {
  passed: boolean;
  duration: number;
  errors: string[];
  screenshots: string[];
}

describe('Profile Update - Cross-Platform Parity', () => {
  const testCases: PlatformTestSuite = {
    web: async () => {
      const page = await browser.newPage();
      await page.goto('/profile/edit');

      // Test validation
      await page.fill('[name="displayName"]', 'A'.repeat(101));
      await expect(page.locator('.error-message')).toContainText('100 characters or less');

      // Test character counter
      await page.fill('[name="bio"]', 'Test bio');
      await expect(page.locator('.char-counter')).toContainText('8/500');

      return {
        passed: true,
        duration: Date.now(),
        errors: [],
        screenshots: []
      };
    },

    ios: async () => {
      await device.launchApp();
      await element(by.id('profile-edit-button')).tap();

      // Test validation
      await element(by.id('display-name-field')).typeText('A'.repeat(101));
      await expect(element(by.id('error-message'))).toHaveText('100 characters or less');

      return {
        passed: true,
        duration: Date.now(),
        errors: [],
        screenshots: []
      };
    },

    android: async () => {
      await device.launchApp();
      await element(by.id('profile_edit_button')).tap();

      // Test validation
      await element(by.id('display_name_field')).typeText('A'.repeat(101));
      await expect(element(by.id('error_message'))).toHaveText('100 characters or less');

      return {
        passed: true,
        duration: Date.now(),
        errors: [],
        screenshots: []
      };
    },

    desktop: async () => {
      // Similar to web test
      return {
        passed: true,
        duration: Date.now(),
        errors: [],
        screenshots: []
      };
    }
  };

  test('validates display name length across all platforms', async () => {
    const results = await Promise.all([
      testCases.web(),
      testCases.ios(),
      testCases.android(),
      testCases.desktop()
    ]);

    // Verify all platforms pass
    results.forEach((result, index) => {
      expect(result.passed).toBe(true);
    });

    // Verify variance in duration is acceptable (<20%)
    const durations = results.map(r => r.duration);
    const avgDuration = durations.reduce((a, b) => a + b) / durations.length;
    const maxVariance = Math.max(...durations) - Math.min(...durations);

    expect(maxVariance / avgDuration).toBeLessThan(0.2);
  });
});
```

---

## Platform Testing Frameworks

### Web Testing with Playwright

**Setup:**

```bash
npm install -D @playwright/test
npx playwright install
```

**Configuration:**

```typescript
// playwright.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html'],
    ['json', { outputFile: 'test-results.json' }]
  ],
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure'
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] }
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] }
    },
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] }
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] }
    }
  ],

  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI
  }
});
```

**Test Examples:**

```typescript
// tests/e2e/profile.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Profile Management', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'password123');
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard');
  });

  test('should update profile successfully', async ({ page }) => {
    await page.goto('/profile/edit');

    // Fill form
    await page.fill('[name="displayName"]', 'John Doe');
    await page.fill('[name="bio"]', 'Software engineer');

    // Submit
    await page.click('button[type="submit"]');

    // Verify success
    await expect(page.locator('.success-message')).toBeVisible();
    await expect(page.locator('.profile-name')).toHaveText('John Doe');
  });

  test('should validate required fields', async ({ page }) => {
    await page.goto('/profile/edit');

    // Clear required field
    await page.fill('[name="displayName"]', '');

    // Attempt submit
    await page.click('button[type="submit"]');

    // Verify error
    await expect(page.locator('.error-message')).toContainText('required');
  });

  test('should enforce character limits', async ({ page }) => {
    await page.goto('/profile/edit');

    // Type beyond limit
    await page.fill('[name="bio"]', 'A'.repeat(501));

    // Verify counter and error
    await expect(page.locator('.char-counter')).toContainText('501/500');
    await expect(page.locator('[name="bio"]')).toHaveAttribute('aria-invalid', 'true');
  });

  test('should handle network errors gracefully', async ({ page, context }) => {
    await context.route('**/api/users/*', route => route.abort());

    await page.goto('/profile/edit');
    await page.fill('[name="displayName"]', 'Jane Doe');
    await page.click('button[type="submit"]');

    // Verify error handling
    await expect(page.locator('.error-message')).toContainText('network');
  });
});
```

### iOS Testing with XCTest and XCUITest

**Unit Tests:**

```swift
// Tests/ProfileViewModelTests.swift
import XCTest
@testable import MyApp

final class ProfileViewModelTests: XCTestCase {
    var viewModel: ProfileViewModel!
    var mockRepository: MockUserRepository!

    override func setUp() {
        super.setUp()
        mockRepository = MockUserRepository()
        viewModel = ProfileViewModel(
            userId: "test-user",
            repository: mockRepository
        )
    }

    override func tearDown() {
        viewModel = nil
        mockRepository = nil
        super.tearDown()
    }

    func testValidateDisplayName_WithValidInput_ReturnsTrue() {
        // Arrange
        viewModel.displayName = "John Doe"

        // Act
        let result = viewModel.validate()

        // Assert
        XCTAssertTrue(result)
        XCTAssertNil(viewModel.displayNameError)
    }

    func testValidateDisplayName_WithTooLongInput_ReturnsFalse() {
        // Arrange
        viewModel.displayName = String(repeating: "A", count: 101)

        // Act
        let result = viewModel.validate()

        // Assert
        XCTAssertFalse(result)
        XCTAssertEqual(
            viewModel.displayNameError,
            "Display name must be 100 characters or less"
        )
    }

    func testUpdateProfile_WithValidData_CallsRepository() async {
        // Arrange
        viewModel.displayName = "John Doe"
        viewModel.bio = "Software engineer"
        let expectedProfile = UserProfile(
            id: "test-user",
            username: "johndoe",
            email: "john@example.com",
            displayName: "John Doe",
            bio: "Software engineer",
            // ... other fields
        )
        mockRepository.updateProfileResult = .success(expectedProfile)

        // Act
        await viewModel.updateProfile()

        // Assert
        XCTAssertTrue(mockRepository.updateProfileCalled)
        XCTAssertEqual(viewModel.profile?.displayName, "John Doe")
        XCTAssertFalse(viewModel.isLoading)
        XCTAssertNil(viewModel.error)
    }

    func testUpdateProfile_WithNetworkError_SetsErrorMessage() async {
        // Arrange
        viewModel.displayName = "John Doe"
        mockRepository.updateProfileResult = .failure(.networkError("Connection failed"))

        // Act
        await viewModel.updateProfile()

        // Assert
        XCTAssertNotNil(viewModel.error)
        XCTAssertFalse(viewModel.isLoading)
    }
}

// Mock Repository
class MockUserRepository: UserRepository {
    var updateProfileCalled = false
    var updateProfileResult: Result<UserProfile, RepositoryError>?

    func updateProfile(
        userId: String,
        displayName: String?,
        bio: String?
    ) async -> Result<UserProfile, RepositoryError> {
        updateProfileCalled = true
        return updateProfileResult ?? .failure(.unknown)
    }
}
```

**UI Tests:**

```swift
// UITests/ProfileUITests.swift
import XCTest

final class ProfileUITests: XCTestCase {
    var app: XCUIApplication!

    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = ["UITesting"]
        app.launch()

        // Login
        let emailField = app.textFields["email-field"]
        emailField.tap()
        emailField.typeText("test@example.com")

        let passwordField = app.secureTextFields["password-field"]
        passwordField.tap()
        passwordField.typeText("password123")

        app.buttons["login-button"].tap()
    }

    func testUpdateProfile_WithValidData_ShowsSuccessMessage() {
        // Navigate to profile edit
        app.buttons["profile-button"].tap()
        app.buttons["edit-button"].tap()

        // Fill form
        let displayNameField = app.textFields["display-name-field"]
        displayNameField.tap()
        displayNameField.clearAndType(text: "John Doe")

        let bioField = app.textViews["bio-field"]
        bioField.tap()
        bioField.clearAndType(text: "Software engineer")

        // Submit
        app.buttons["save-button"].tap()

        // Verify success
        XCTAssertTrue(app.staticTexts["success-message"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["John Doe"].exists)
    }

    func testUpdateProfile_WithInvalidDisplayName_ShowsError() {
        app.buttons["profile-button"].tap()
        app.buttons["edit-button"].tap()

        let displayNameField = app.textFields["display-name-field"]
        displayNameField.tap()
        displayNameField.clearAndType(text: String(repeating: "A", count: 101))

        app.buttons["save-button"].tap()

        XCTAssertTrue(app.staticTexts["error-message"].exists)
        XCTAssertTrue(app.staticTexts["error-message"].label.contains("100 characters"))
    }

    func testCharacterCounter_UpdatesInRealTime() {
        app.buttons["profile-button"].tap()
        app.buttons["edit-button"].tap()

        let bioField = app.textViews["bio-field"]
        bioField.tap()
        bioField.typeText("Test")

        let counter = app.staticTexts["char-counter"]
        XCTAssertTrue(counter.label.contains("4/500"))
    }
}

extension XCUIElement {
    func clearAndType(text: String) {
        guard let stringValue = self.value as? String else {
            XCTFail("Tried to clear and type on a non string value")
            return
        }

        self.tap()
        let deleteString = String(repeating: XCUIKeyboardKey.delete.rawValue, count: stringValue.count)
        self.typeText(deleteString)
        self.typeText(text)
    }
}
```

### Android Testing with Espresso and Compose

**Unit Tests:**

```kotlin
// ProfileViewModelTest.kt
package com.example.ui.profile

import androidx.arch.core.executor.testing.InstantTaskExecutorRule
import com.example.data.models.UserProfile
import com.example.data.repository.Result
import com.example.data.repository.UserRepository
import io.mockk.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.*
import org.junit.*
import org.junit.Assert.*

@OptIn(ExperimentalCoroutinesApi::class)
class ProfileViewModelTest {
    @get:Rule
    val instantExecutorRule = InstantTaskExecutorRule()

    private lateinit var viewModel: ProfileViewModel
    private lateinit var repository: UserRepository
    private val testDispatcher = StandardTestDispatcher()

    @Before
    fun setup() {
        Dispatchers.setMain(testDispatcher)
        repository = mockk()
        viewModel = ProfileViewModel(repository)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `loadProfile should update uiState with profile data`() = runTest {
        // Arrange
        val userId = "test-user"
        val expectedProfile = UserProfile(
            id = userId,
            username = "johndoe",
            email = "john@example.com",
            displayName = "John Doe",
            // ... other fields
        )

        coEvery { repository.getProfile(userId) } returns flowOf(
            Result.Success(expectedProfile)
        )

        // Act
        viewModel.loadProfile(userId)
        advanceUntilIdle()

        // Assert
        assertEquals(expectedProfile, viewModel.uiState.value.profile)
        assertFalse(viewModel.uiState.value.isLoading)
        assertNull(viewModel.uiState.value.error)
    }

    @Test
    fun `updateProfile with valid data should succeed`() = runTest {
        // Arrange
        val userId = "test-user"
        viewModel.updateDisplayName("John Doe")
        viewModel.updateBio("Software engineer")

        val expectedProfile = UserProfile(
            id = userId,
            username = "johndoe",
            email = "john@example.com",
            displayName = "John Doe",
            bio = "Software engineer",
            // ... other fields
        )

        coEvery {
            repository.updateProfile(userId, "John Doe", "Software engineer")
        } returns Result.Success(expectedProfile)

        // Act
        viewModel.updateProfile(userId)
        advanceUntilIdle()

        // Assert
        assertEquals(expectedProfile, viewModel.uiState.value.profile)
        assertFalse(viewModel.uiState.value.isLoading)
        coVerify {
            repository.updateProfile(userId, "John Doe", "Software engineer")
        }
    }

    @Test
    fun `validate should return false for displayName exceeding 100 characters`() {
        // Arrange
        viewModel.updateDisplayName("A".repeat(101))

        // Act (validate is called internally during updateProfile)
        // We need to trigger validation
        viewModel.updateProfile("test-user")

        // Assert
        assertEquals(
            "Display name must be 100 characters or less",
            viewModel.uiState.value.displayNameError
        )
    }
}
```

**Compose UI Tests:**

```kotlin
// ProfileScreenTest.kt
package com.example.ui.profile

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.data.models.*
import io.mockk.*
import org.junit.*
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ProfileScreenTest {
    @get:Rule
    val composeTestRule = createComposeRule()

    private lateinit var viewModel: ProfileViewModel

    @Before
    fun setup() {
        viewModel = mockk(relaxed = true)
    }

    @Test
    fun profileScreen_displaysProfileData() {
        // Arrange
        val profile = UserProfile(
            id = "test-user",
            username = "johndoe",
            email = "john@example.com",
            displayName = "John Doe",
            bio = "Software engineer",
            // ... other fields
        )

        every { viewModel.uiState } returns MutableStateFlow(
            ProfileUiState(profile = profile)
        )

        // Act
        composeTestRule.setContent {
            ProfileScreen(userId = "test-user", viewModel = viewModel) {}
        }

        // Assert
        composeTestRule.onNodeWithText("John Doe").assertExists()
        composeTestRule.onNodeWithText("Software engineer").assertExists()
    }

    @Test
    fun profileScreen_showsCharacterCounter() {
        every { viewModel.uiState } returns MutableStateFlow(
            ProfileUiState(bio = "Test bio")
        )

        composeTestRule.setContent {
            ProfileScreen(userId = "test-user", viewModel = viewModel) {}
        }

        composeTestRule.onNodeWithText("8/500").assertExists()
    }

    @Test
    fun profileScreen_displaysErrorForLongBio() {
        val longBio = "A".repeat(501)

        every { viewModel.uiState } returns MutableStateFlow(
            ProfileUiState(
                bio = longBio,
                bioError = "Bio must be 500 characters or less"
            )
        )

        composeTestRule.setContent {
            ProfileScreen(userId = "test-user", viewModel = viewModel) {}
        }

        composeTestRule.onNodeWithText("Bio must be 500 characters or less")
            .assertExists()
    }

    @Test
    fun saveButton_triggersUpdateProfile() {
        every { viewModel.uiState } returns MutableStateFlow(ProfileUiState())

        composeTestRule.setContent {
            ProfileScreen(userId = "test-user", viewModel = viewModel) {}
        }

        // Type in fields
        composeTestRule.onNodeWithText("Display Name")
            .performTextInput("John Doe")

        // Click save
        composeTestRule.onNodeWithText("Save").performClick()

        // Verify viewModel was called
        verify { viewModel.updateProfile("test-user") }
    }
}
```

### Mobile Testing with Maestro

**Installation:**

```bash
curl -Ls "https://get.maestro.mobile.dev" | bash
```

**Flow Configuration:**

```yaml
# maestro/profileUpdate.yaml
appId: com.example.myapp
---
- launchApp
- tapOn: "Profile"
- tapOn: "Edit"

# Test display name validation
- inputText:
    text: "John Doe"
- tapOn: "Save"
- assertVisible: "Profile updated successfully"

# Test character limit
- tapOn: "Edit"
- clearState
- inputText:
    text: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" # 101 chars
- assertVisible: "Display name must be 100 characters or less"

# Test bio character counter
- tapOn:
    id: "bio-field"
- inputText: "Test bio"
- assertVisible: "8/500"
```

**Run Tests:**

```bash
maestro test maestro/profileUpdate.yaml
maestro test --format junit maestro/profileUpdate.yaml
```

---

## Performance Benchmarking

### Web Performance (Lighthouse)

**Automated Lighthouse CI:**

```javascript
// lighthouserc.js
module.exports = {
  ci: {
    collect: {
      url: [
        'http://localhost:3000/',
        'http://localhost:3000/profile',
        'http://localhost:3000/profile/edit'
      ],
      numberOfRuns: 5,
      settings: {
        preset: 'desktop'
      }
    },
    assert: {
      assertions: {
        'categories:performance': ['error', { minScore: 0.9 }],
        'categories:accessibility': ['error', { minScore: 0.95 }],
        'categories:best-practices': ['error', { minScore: 0.9 }],
        'categories:seo': ['error', { minScore: 0.9 }],
        'first-contentful-paint': ['error', { maxNumericValue: 2000 }],
        'largest-contentful-paint': ['error', { maxNumericValue: 2500 }],
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
        'total-blocking-time': ['error', { maxNumericValue: 300 }]
      }
    },
    upload: {
      target: 'temporary-public-storage'
    }
  }
};
```

### iOS Performance (XCTest Metrics)

```swift
// Tests/PerformanceTests.swift
import XCTest
@testable import MyApp

final class ProfilePerformanceTests: XCTestCase {
    var viewModel: ProfileViewModel!

    override func setUp() {
        super.setUp()
        viewModel = ProfileViewModel(userId: "test-user")
    }

    func testProfileLoadPerformance() throws {
        measure(metrics: [XCTClockMetric(), XCTMemoryMetric()]) {
            let expectation = XCTestExpectation(description: "Profile loaded")

            Task {
                await viewModel.loadProfile()
                expectation.fulfill()
            }

            wait(for: [expectation], timeout: 2.0)
        }
    }

    func testScrollPerformance() throws {
        let app = XCUIApplication()
        app.launch()

        app.buttons["feed-tab"].tap()

        let options = XCTMeasureOptions()
        options.invocationOptions = [.manuallyStop]

        measure(metrics: [XCTOSSignpostMetric.scrollDecelerationMetric], options: options) {
            let table = app.tables.firstMatch
            table.swipeUp(velocity: .fast)

            // Scroll until we reach bottom
            var previousCellCount = 0
            while true {
                let cellCount = table.cells.count
                if cellCount == previousCellCount {
                    break
                }
                previousCellCount = cellCount
                table.swipeUp()
            }

            stopMeasuring()
        }
    }

    func testMemoryLeaks() throws {
        measure(metrics: [XCTMemoryMetric()]) {
            for _ in 0..<100 {
                let vm = ProfileViewModel(userId: "test-user")
                _ = vm
            }
        }
    }
}
```

### Android Performance (Macrobenchmark)

```kotlin
// benchmark/ProfileBenchmark.kt
package com.example.benchmark

import androidx.benchmark.macro.*
import androidx.benchmark.macro.junit4.MacrobenchmarkRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.uiautomator.By
import androidx.test.uiautomator.Until
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ProfileBenchmark {
    @get:Rule
    val benchmarkRule = MacrobenchmarkRule()

    @Test
    fun startupCold() = benchmarkRule.measureRepeated(
        packageName = "com.example.myapp",
        metrics = listOf(StartupTimingMetric()),
        iterations = 5,
        startupMode = StartupMode.COLD
    ) {
        pressHome()
        startActivityAndWait()
    }

    @Test
    fun startupWarm() = benchmarkRule.measureRepeated(
        packageName = "com.example.myapp",
        metrics = listOf(StartupTimingMetric()),
        iterations = 5,
        startupMode = StartupMode.WARM
    ) {
        pressHome()
        startActivityAndWait()
    }

    @Test
    fun profileScroll() = benchmarkRule.measureRepeated(
        packageName = "com.example.myapp",
        metrics = listOf(FrameTimingMetric()),
        iterations = 5,
        setupBlock = {
            pressHome()
            startActivityAndWait()

            // Navigate to profile
            device.findObject(By.res("profile_button")).click()
            device.wait(Until.hasObject(By.res("profile_content")), 2000)
        }
    ) {
        val list = device.findObject(By.res("profile_feed"))
        list.setGestureMargin(device.displayWidth / 5)
        list.fling(Direction.DOWN)
        device.waitForIdle()
    }

    @Test
    fun profileEdit() = benchmarkRule.measureRepeated(
        packageName = "com.example.myapp",
        metrics = listOf(
            StartupTimingMetric(),
            FrameTimingMetric()
        ),
        iterations = 5,
        setupBlock = {
            pressHome()
            startActivityAndWait()
        }
    ) {
        // Navigate to edit screen
        device.findObject(By.res("profile_button")).click()
        device.findObject(By.res("edit_button")).click()

        // Fill form
        device.findObject(By.res("display_name_field")).text = "John Doe"
        device.findObject(By.res("bio_field")).text = "Software engineer"

        // Submit
        device.findObject(By.res("save_button")).click()

        device.wait(Until.hasObject(By.text("Profile updated")), 3000)
    }
}
```

---

## End-to-End Testing

### Multi-Platform E2E with Custom Framework

```typescript
// tests/e2e/framework/MultiPlatformTest.ts
import { PlaywrightTestConfig } from '@playwright/test';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface TestScenario {
  name: string;
  web?: () => Promise<void>;
  ios?: () => Promise<void>;
  android?: () => Promise<void>;
  desktop?: () => Promise<void>;
}

export class MultiPlatformTester {
  async runScenario(scenario: TestScenario): Promise<TestResults> {
    const results: TestResults = {
      web: null,
      ios: null,
      android: null,
      desktop: null
    };

    // Run tests in parallel
    await Promise.all([
      scenario.web ? this.runWebTest(scenario.web).then(r => results.web = r) : Promise.resolve(),
      scenario.ios ? this.runIOSTest(scenario.ios).then(r => results.ios = r) : Promise.resolve(),
      scenario.android ? this.runAndroidTest(scenario.android).then(r => results.android = r) : Promise.resolve(),
      scenario.desktop ? this.runDesktopTest(scenario.desktop).then(r => results.desktop = r) : Promise.resolve()
    ]);

    return results;
  }

  private async runWebTest(test: () => Promise<void>): Promise<PlatformResult> {
    const start = Date.now();
    try {
      await test();
      return {
        passed: true,
        duration: Date.now() - start,
        errors: []
      };
    } catch (error) {
      return {
        passed: false,
        duration: Date.now() - start,
        errors: [error.message]
      };
    }
  }

  private async runIOSTest(test: () => Promise<void>): Promise<PlatformResult> {
    // Launch iOS simulator
    await execAsync('xcrun simctl boot "iPhone 14"');

    const start = Date.now();
    try {
      await test();
      return {
        passed: true,
        duration: Date.now() - start,
        errors: []
      };
    } catch (error) {
      return {
        passed: false,
        duration: Date.now() - start,
        errors: [error.message]
      };
    }
  }

  private async runAndroidTest(test: () => Promise<void>): Promise<PlatformResult> {
    // Launch Android emulator
    await execAsync('adb shell am start -n com.example.myapp/.MainActivity');

    const start = Date.now();
    try {
      await test();
      return {
        passed: true,
        duration: Date.now() - start,
        errors: []
      };
    } catch (error) {
      return {
        passed: false,
        duration: Date.now() - start,
        errors: [error.message]
      };
    }
  }
}
```

---

## Visual Regression Testing

### Percy for Multi-Platform Screenshots

```typescript
// tests/visual/profile.visual.ts
import percySnapshot from '@percy/playwright';
import { test } from '@playwright/test';

test.describe('Profile Visual Regression', () => {
  test('profile edit form - desktop', async ({ page }) => {
    await page.goto('/profile/edit');
    await percySnapshot(page, 'Profile Edit - Desktop');
  });

  test('profile edit form - mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto('/profile/edit');
    await percySnapshot(page, 'Profile Edit - Mobile');
  });

  test('profile edit form - error state', async ({ page }) => {
    await page.goto('/profile/edit');
    await page.fill('[name="displayName"]', 'A'.repeat(101));
    await page.click('button[type="submit"]');
    await percySnapshot(page, 'Profile Edit - Error State');
  });
});
```

---

This testing strategies guide provides comprehensive patterns for validating feature parity, automating tests across platforms, benchmarking performance, and ensuring consistent behavior in multi-platform applications.
