---
name: mobile-testing-patterns
description: "Test mobile applications across iOS and Android with Detox, Maestro, and Appium including device testing, screenshot testing, performance profiling, and CI integration for mobile. Use when testing React Native, Flutter, or native mobile apps, or setting up mobile CI pipelines."
---

# Mobile Testing Patterns

## Expert Agent

For mobile testing strategy, test automation, and quality assurance, delegate to:

- **`quality-specialist`**: Expert in software quality through code reviews, security audits, and test automation.
  - *Location*: `plugins/dev-suite/agents/quality-specialist.md`

## Detox (React Native)

### Configuration (.detoxrc.js)

```javascript
module.exports = {
  testRunner: { args: { $0: 'jest', config: 'e2e/jest.config.js' } },
  apps: {
    'ios.debug': {
      type: 'ios.app',
      binaryPath: 'ios/build/Build/Products/Debug-iphonesimulator/MyApp.app',
      build: 'xcodebuild -workspace ios/MyApp.xcworkspace -scheme MyApp -configuration Debug -sdk iphonesimulator',
    },
    'android.debug': {
      type: 'android.apk',
      binaryPath: 'android/app/build/outputs/apk/debug/app-debug.apk',
      build: 'cd android && ./gradlew assembleDebug assembleAndroidTest',
    },
  },
  devices: {
    simulator: { type: 'ios.simulator', device: { type: 'iPhone 15' } },
    emulator: { type: 'android.emulator', device: { avdName: 'Pixel_7_API_34' } },
  },
  configurations: {
    'ios.sim.debug': { device: 'simulator', app: 'ios.debug' },
    'android.emu.debug': { device: 'emulator', app: 'android.debug' },
  },
};
```

### Writing Tests

```javascript
describe('Login Flow', () => {
  beforeAll(async () => { await device.launchApp({ newInstance: true }); });
  beforeEach(async () => { await device.reloadReactNative(); });

  it('should login with valid credentials', async () => {
    await element(by.id('email-input')).typeText('user@example.com');
    await element(by.id('password-input')).typeText('secure123');
    await element(by.id('login-button')).tap();
    await waitFor(element(by.id('home-screen'))).toBeVisible().withTimeout(5000);
    await expect(element(by.text('Welcome'))).toBeVisible();
  });
});
```

## Maestro (Cross-Platform YAML Flows)

```yaml
appId: com.myapp
---
- launchApp
- tapOn:
    id: "email-input"
- inputText: "user@example.com"
- tapOn:
    id: "password-input"
- inputText: "secure123"
- tapOn: "Login"
- assertVisible: "Welcome"
```

```bash
maestro test flows/login.yaml
maestro test flows/                  # run entire suite
maestro cloud flows/ --app-file app.apk  # cloud execution
```

## Appium (Native + Hybrid)

```python
from appium import webdriver
from appium.options.android import UiAutomator2Options

options = UiAutomator2Options()
options.platform_name = "Android"
options.device_name = "Pixel_7"
options.app = "/path/to/app.apk"
options.automation_name = "UiAutomator2"

driver = webdriver.Remote("http://localhost:4723", options=options)

class LoginPage:
    """Page object for login screen."""
    def __init__(self, driver):
        self.driver = driver

    def login(self, email: str, password: str):
        self.driver.find_element("accessibility id", "email-input").send_keys(email)
        self.driver.find_element("accessibility id", "password-input").send_keys(password)
        self.driver.find_element("accessibility id", "login-button").click()
```

## Device Farms (CI Integration)

### Firebase Test Lab

```yaml
# .github/workflows/mobile-test.yml
jobs:
  android-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cd android && ./gradlew assembleDebug assembleAndroidTest
      - uses: google-github-actions/auth@v2
        with: { credentials_json: "${{ secrets.GCP_SA_KEY }}" }
      - run: |
          gcloud firebase test android run \
            --type instrumentation \
            --app app-debug.apk --test app-debug-androidTest.apk \
            --device model=Pixel7,version=34 --timeout 10m
```

### AWS Device Farm

```bash
aws devicefarm schedule-run \
  --project-arn $PROJECT_ARN --app-arn $APP_ARN \
  --device-pool-arn $POOL_ARN \
  --test '{"type":"APPIUM_PYTHON","testPackageArn":"'$TEST_ARN'"}'
```

## Screenshot Testing

```javascript
it('should match login screen snapshot', async () => {
  await device.takeScreenshot('login-screen');
});
```

## Framework Selection

| Criterion | Detox | Maestro | Appium |
|-----------|-------|---------|--------|
| Best for | React Native | Quick flows | Native/Hybrid |
| Setup | Medium | Low | High |
| Speed | Fast (gray-box) | Fast | Slower (black-box) |
| Language | JavaScript | YAML | Python/Java/JS |
| Flakiness | Low | Low | Medium |

## Mobile CI Checklist

- [ ] Build both debug and release variants for testing
- [ ] Run unit tests before E2E tests (fail fast)
- [ ] Use device farms for cross-device coverage (min 3 Android, 2 iOS)
- [ ] Capture screenshots on test failure for debugging
- [ ] Set timeouts: 120s per test, 30min per suite
- [ ] Cache Gradle/CocoaPods dependencies in CI
- [ ] Store test artifacts (screenshots, logs, videos)
- [ ] Gate releases on E2E pass rate (target 95%+)
