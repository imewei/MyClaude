# Design Systems Guide

Cross-platform design system implementation covering design tokens, Material Design 3, iOS Human Interface Guidelines, component libraries, and accessibility standards.

## Table of Contents

1. [Cross-Platform Design Tokens](#cross-platform-design-tokens)
2. [Material Design 3 (Android/Web)](#material-design-3-androidweb)
3. [iOS Human Interface Guidelines](#ios-human-interface-guidelines)
4. [Component Libraries](#component-libraries)
5. [Accessibility Standards](#accessibility-standards)

---

## Cross-Platform Design Tokens

### Design Token Structure

**Token Hierarchy:**

```
Design Tokens
├── Primitives (color palettes, base spacing)
├── Semantic (brand, surface, text)
└── Component (button-primary, card-shadow)
```

### Color System

**JSON Format (Platform-Agnostic):**

```json
{
  "color": {
    "primitives": {
      "blue": {
        "50": "#e3f2fd",
        "100": "#bbdefb",
        "200": "#90caf9",
        "300": "#64b5f6",
        "400": "#42a5f5",
        "500": "#2196f3",
        "600": "#1e88e5",
        "700": "#1976d2",
        "800": "#1565c0",
        "900": "#0d47a1"
      },
      "gray": {
        "50": "#fafafa",
        "100": "#f5f5f5",
        "200": "#eeeeee",
        "300": "#e0e0e0",
        "400": "#bdbdbd",
        "500": "#9e9e9e",
        "600": "#757575",
        "700": "#616161",
        "800": "#424242",
        "900": "#212121"
      }
    },
    "semantic": {
      "light": {
        "primary": "{color.primitives.blue.600}",
        "primary-variant": "{color.primitives.blue.700}",
        "secondary": "{color.primitives.purple.500}",
        "background": "{color.primitives.gray.50}",
        "surface": "#ffffff",
        "error": "#b00020",
        "on-primary": "#ffffff",
        "on-secondary": "#ffffff",
        "on-background": "{color.primitives.gray.900}",
        "on-surface": "{color.primitives.gray.900}",
        "on-error": "#ffffff"
      },
      "dark": {
        "primary": "{color.primitives.blue.400}",
        "primary-variant": "{color.primitives.blue.300}",
        "secondary": "{color.primitives.purple.300}",
        "background": "#121212",
        "surface": "#1e1e1e",
        "error": "#cf6679",
        "on-primary": "{color.primitives.gray.900}",
        "on-secondary": "{color.primitives.gray.900}",
        "on-background": "#ffffff",
        "on-surface": "#ffffff",
        "on-error": "{color.primitives.gray.900}"
      }
    },
    "component": {
      "button": {
        "primary": {
          "background": "{color.semantic.light.primary}",
          "text": "{color.semantic.light.on-primary}",
          "hover": "{color.primitives.blue.700}",
          "active": "{color.primitives.blue.800}",
          "disabled-background": "{color.primitives.gray.300}",
          "disabled-text": "{color.primitives.gray.500}"
        }
      }
    }
  }
}
```

**CSS Custom Properties:**

```css
/* tokens.css */
:root {
  /* Primitives */
  --color-blue-500: #2196f3;
  --color-blue-600: #1e88e5;
  --color-blue-700: #1976d2;

  /* Semantic - Light Theme */
  --color-primary: var(--color-blue-600);
  --color-primary-variant: var(--color-blue-700);
  --color-background: #fafafa;
  --color-surface: #ffffff;
  --color-on-primary: #ffffff;
  --color-on-background: #212121;

  /* Component Tokens */
  --button-primary-bg: var(--color-primary);
  --button-primary-text: var(--color-on-primary);
  --button-primary-hover: var(--color-blue-700);
}

[data-theme="dark"] {
  --color-primary: #42a5f5;
  --color-background: #121212;
  --color-surface: #1e1e1e;
  --color-on-primary: #212121;
  --color-on-background: #ffffff;
}

/* Usage */
.button-primary {
  background-color: var(--button-primary-bg);
  color: var(--button-primary-text);
}

.button-primary:hover {
  background-color: var(--button-primary-hover);
}
```

**Swift (iOS):**

```swift
// DesignTokens/Colors.swift
import SwiftUI

extension Color {
    // Primitives
    static let blue500 = Color(hex: "#2196f3")
    static let blue600 = Color(hex: "#1e88e5")
    static let blue700 = Color(hex: "#1976d2")

    // Semantic Tokens
    static let brandPrimary = Color("BrandPrimary") // Asset catalog
    static let brandSecondary = Color("BrandSecondary")
    static let appBackground = Color("AppBackground")
    static let appSurface = Color("AppSurface")

    // Component Tokens
    struct Button {
        static let primaryBackground = Color.brandPrimary
        static let primaryText = Color.white
        static let primaryHover = Color.blue700
    }
}

// Asset Catalog (Colors.xcassets)
// BrandPrimary
//   - Light Appearance: #1e88e5
//   - Dark Appearance: #42a5f5
```

**Kotlin (Android):**

```kotlin
// ui/theme/Color.kt
package com.example.ui.theme

import androidx.compose.ui.graphics.Color

// Primitives
val Blue500 = Color(0xFF2196F3)
val Blue600 = Color(0xFF1E88E5)
val Blue700 = Color(0xFF1976D2)
val Gray50 = Color(0xFFFAFAFA)
val Gray900 = Color(0xFF212121)

// Semantic - Light
val LightPrimary = Blue600
val LightBackground = Gray50
val LightSurface = Color.White
val LightOnPrimary = Color.White

// Semantic - Dark
val DarkPrimary = Color(0xFF42A5F5)
val DarkBackground = Color(0xFF121212)
val DarkSurface = Color(0xFF1E1E1E)
val DarkOnPrimary = Gray900
```

### Typography System

**Scale Definition:**

```json
{
  "typography": {
    "font-family": {
      "sans": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      "serif": "Georgia, serif",
      "mono": "'JetBrains Mono', 'Fira Code', monospace"
    },
    "font-size": {
      "xs": "0.75rem",    // 12px
      "sm": "0.875rem",   // 14px
      "base": "1rem",     // 16px
      "lg": "1.125rem",   // 18px
      "xl": "1.25rem",    // 20px
      "2xl": "1.5rem",    // 24px
      "3xl": "1.875rem",  // 30px
      "4xl": "2.25rem",   // 36px
      "5xl": "3rem"       // 48px
    },
    "font-weight": {
      "light": 300,
      "normal": 400,
      "medium": 500,
      "semibold": 600,
      "bold": 700
    },
    "line-height": {
      "tight": 1.25,
      "normal": 1.5,
      "relaxed": 1.75
    },
    "text-styles": {
      "h1": {
        "font-size": "{typography.font-size.5xl}",
        "font-weight": "{typography.font-weight.bold}",
        "line-height": "{typography.line-height.tight}",
        "letter-spacing": "-0.025em"
      },
      "h2": {
        "font-size": "{typography.font-size.4xl}",
        "font-weight": "{typography.font-weight.bold}",
        "line-height": "{typography.line-height.tight}"
      },
      "body": {
        "font-size": "{typography.font-size.base}",
        "font-weight": "{typography.font-weight.normal}",
        "line-height": "{typography.line-height.normal}"
      },
      "caption": {
        "font-size": "{typography.font-size.sm}",
        "font-weight": "{typography.font-weight.normal}",
        "line-height": "{typography.line-height.normal}"
      }
    }
  }
}
```

**Tailwind Configuration:**

```javascript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', ...defaultTheme.fontFamily.sans],
      },
      fontSize: {
        xs: '0.75rem',
        sm: '0.875rem',
        base: '1rem',
        lg: '1.125rem',
        xl: '1.25rem',
        '2xl': '1.5rem',
        '3xl': '1.875rem',
        '4xl': '2.25rem',
        '5xl': '3rem',
      },
      fontWeight: {
        light: 300,
        normal: 400,
        medium: 500,
        semibold: 600,
        bold: 700,
      }
    }
  }
};
```

### Spacing System

**8pt Grid System:**

```json
{
  "spacing": {
    "0": "0",
    "1": "0.25rem",   // 4px
    "2": "0.5rem",    // 8px
    "3": "0.75rem",   // 12px
    "4": "1rem",      // 16px
    "5": "1.25rem",   // 20px
    "6": "1.5rem",    // 24px
    "8": "2rem",      // 32px
    "10": "2.5rem",   // 40px
    "12": "3rem",     // 48px
    "16": "4rem",     // 64px
    "20": "5rem",     // 80px
    "24": "6rem"      // 96px
  }
}
```

**SwiftUI:**

```swift
// DesignTokens/Spacing.swift
extension CGFloat {
    static let spacing1: CGFloat = 4
    static let spacing2: CGFloat = 8
    static let spacing3: CGFloat = 12
    static let spacing4: CGFloat = 16
    static let spacing5: CGFloat = 20
    static let spacing6: CGFloat = 24
    static let spacing8: CGFloat = 32
    static let spacing10: CGFloat = 40
    static let spacing12: CGFloat = 48
}

// Usage
VStack(spacing: .spacing4) {
    Text("Hello")
    Text("World")
}
.padding(.spacing6)
```

**Jetpack Compose:**

```kotlin
// ui/theme/Spacing.kt
package com.example.ui.theme

import androidx.compose.ui.unit.dp

object Spacing {
    val spacing1 = 4.dp
    val spacing2 = 8.dp
    val spacing3 = 12.dp
    val spacing4 = 16.dp
    val spacing5 = 20.dp
    val spacing6 = 24.dp
    val spacing8 = 32.dp
    val spacing10 = 40.dp
    val spacing12 = 48.dp
}

// Usage
Column(
    modifier = Modifier.padding(Spacing.spacing6),
    verticalArrangement = Arrangement.spacedBy(Spacing.spacing4)
) {
    Text("Hello")
    Text("World")
}
```

---

## Material Design 3 (Android/Web)

### Material You Dynamic Theming

**Dynamic Color Scheme:**

```kotlin
// ui/theme/Theme.kt
package com.example.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val LightColorScheme = lightColorScheme(
    primary = Blue600,
    onPrimary = Color.White,
    primaryContainer = Blue100,
    onPrimaryContainer = Blue900,
    secondary = Purple500,
    onSecondary = Color.White,
    secondaryContainer = Purple100,
    onSecondaryContainer = Purple900,
    tertiary = Teal500,
    onTertiary = Color.White,
    error = Red600,
    onError = Color.White,
    background = Gray50,
    onBackground = Gray900,
    surface = Color.White,
    onSurface = Gray900,
    surfaceVariant = Gray100,
    onSurfaceVariant = Gray700,
    outline = Gray400
)

private val DarkColorScheme = darkColorScheme(
    primary = Blue400,
    onPrimary = Gray900,
    primaryContainer = Blue700,
    onPrimaryContainer = Blue100,
    secondary = Purple300,
    onSecondary = Gray900,
    secondaryContainer = Purple700,
    onSecondaryContainer = Purple100,
    tertiary = Teal300,
    onTertiary = Gray900,
    error = Red400,
    onError = Gray900,
    background = Color(0xFF121212),
    onBackground = Color.White,
    surface = Color(0xFF1E1E1E),
    onSurface = Color.White,
    surfaceVariant = Gray800,
    onSurfaceVariant = Gray300,
    outline = Gray600
)

@Composable
fun MyAppTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context)
            else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.primary.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        shapes = Shapes,
        content = content
    )
}
```

### Material 3 Typography

```kotlin
// ui/theme/Type.kt
package com.example.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

val InterFontFamily = FontFamily(
    Font(R.font.inter_light, FontWeight.Light),
    Font(R.font.inter_regular, FontWeight.Normal),
    Font(R.font.inter_medium, FontWeight.Medium),
    Font(R.font.inter_semibold, FontWeight.SemiBold),
    Font(R.font.inter_bold, FontWeight.Bold)
)

val Typography = Typography(
    displayLarge = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Normal,
        fontSize = 57.sp,
        lineHeight = 64.sp,
        letterSpacing = (-0.25).sp
    ),
    displayMedium = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Normal,
        fontSize = 45.sp,
        lineHeight = 52.sp
    ),
    displaySmall = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Normal,
        fontSize = 36.sp,
        lineHeight = 44.sp
    ),
    headlineLarge = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Normal,
        fontSize = 32.sp,
        lineHeight = 40.sp
    ),
    headlineMedium = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Normal,
        fontSize = 28.sp,
        lineHeight = 36.sp
    ),
    headlineSmall = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Normal,
        fontSize = 24.sp,
        lineHeight = 32.sp
    ),
    titleLarge = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Normal,
        fontSize = 22.sp,
        lineHeight = 28.sp
    ),
    titleMedium = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Medium,
        fontSize = 16.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.15.sp
    ),
    titleSmall = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Medium,
        fontSize = 14.sp,
        lineHeight = 20.sp,
        letterSpacing = 0.1.sp
    ),
    bodyLarge = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.5.sp
    ),
    bodyMedium = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Normal,
        fontSize = 14.sp,
        lineHeight = 20.sp,
        letterSpacing = 0.25.sp
    ),
    bodySmall = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Normal,
        fontSize = 12.sp,
        lineHeight = 16.sp,
        letterSpacing = 0.4.sp
    ),
    labelLarge = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Medium,
        fontSize = 14.sp,
        lineHeight = 20.sp,
        letterSpacing = 0.1.sp
    ),
    labelMedium = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Medium,
        fontSize = 12.sp,
        lineHeight = 16.sp,
        letterSpacing = 0.5.sp
    ),
    labelSmall = TextStyle(
        fontFamily = InterFontFamily,
        fontWeight = FontWeight.Medium,
        fontSize = 11.sp,
        lineHeight = 16.sp,
        letterSpacing = 0.5.sp
    )
)
```

### Material 3 Shapes

```kotlin
// ui/theme/Shape.kt
package com.example.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Shapes
import androidx.compose.ui.unit.dp

val Shapes = Shapes(
    extraSmall = RoundedCornerShape(4.dp),
    small = RoundedCornerShape(8.dp),
    medium = RoundedCornerShape(12.dp),
    large = RoundedCornerShape(16.dp),
    extraLarge = RoundedCornerShape(28.dp)
)
```

### Material 3 Components

**Buttons:**

```kotlin
@Composable
fun MaterialButtonsExample() {
    Column(
        modifier = Modifier.padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // Filled Button (Primary)
        Button(onClick = { /* Action */ }) {
            Text("Filled Button")
        }

        // Filled Tonal Button
        FilledTonalButton(onClick = { /* Action */ }) {
            Text("Filled Tonal")
        }

        // Outlined Button
        OutlinedButton(onClick = { /* Action */ }) {
            Text("Outlined Button")
        }

        // Text Button
        TextButton(onClick = { /* Action */ }) {
            Text("Text Button")
        }

        // Elevated Button
        ElevatedButton(onClick = { /* Action */ }) {
            Text("Elevated Button")
        }
    }
}
```

**Cards:**

```kotlin
@Composable
fun MaterialCardsExample() {
    Column(
        modifier = Modifier.padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // Elevated Card
        ElevatedCard(
            modifier = Modifier.fillMaxWidth(),
            onClick = { /* Action */ }
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Elevated Card", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(8.dp))
                Text("This card has elevation", style = MaterialTheme.typography.bodyMedium)
            }
        }

        // Filled Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer
            )
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Filled Card", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(8.dp))
                Text("This card has a filled background", style = MaterialTheme.typography.bodyMedium)
            }
        }

        // Outlined Card
        OutlinedCard(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Outlined Card", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(8.dp))
                Text("This card has an outline", style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}
```

---

## iOS Human Interface Guidelines

### SF Symbols Integration

```swift
// Views/Components/IconButton.swift
import SwiftUI

struct IconButton: View {
    let systemName: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 20, weight: .medium))
                .foregroundStyle(.tint)
                .frame(width: 44, height: 44) // iOS minimum tap target
        }
    }
}

// Usage with SF Symbols
struct ExampleView: View {
    var body: some View {
        HStack(spacing: 16) {
            IconButton(systemName: "heart") { }
            IconButton(systemName: "heart.fill") { }
            IconButton(systemName: "star") { }
            IconButton(systemName: "star.fill") { }
            IconButton(systemName: "bookmark") { }
            IconButton(systemName: "bookmark.fill") { }
        }
    }
}
```

### Native iOS Typography

```swift
// DesignTokens/Typography.swift
import SwiftUI

extension Font {
    // iOS System Fonts with Dynamic Type
    static let largeTitle = Font.largeTitle
    static let title1 = Font.title
    static let title2 = Font.title2
    static let title3 = Font.title3
    static let headline = Font.headline
    static let body = Font.body
    static let callout = Font.callout
    static let subheadline = Font.subheadline
    static let footnote = Font.footnote
    static let caption1 = Font.caption
    static let caption2 = Font.caption2

    // Custom Fonts with Dynamic Type Support
    static func customLargeTitle() -> Font {
        return .custom("SF Pro Display", size: 34, relativeTo: .largeTitle)
            .weight(.bold)
    }

    static func customHeadline() -> Font {
        return .custom("SF Pro Text", size: 17, relativeTo: .headline)
            .weight(.semibold)
    }
}

// Usage
Text("Large Title")
    .font(.largeTitle)

Text("Custom Headline")
    .font(.customHeadline())
```

### iOS Native Components

**Lists with SwiftUI:**

```swift
// Views/Components/UserList.swift
import SwiftUI

struct UserListView: View {
    let users: [UserProfile]

    var body: some View {
        List(users) { user in
            NavigationLink {
                ProfileDetailView(user: user)
            } label: {
                UserRow(user: user)
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Users")
        .navigationBarTitleDisplayMode(.large)
        .searchable(text: $searchText, prompt: "Search users")
    }
}

struct UserRow: View {
    let user: UserProfile

    var body: some View {
        HStack(spacing: 12) {
            AsyncImage(url: URL(string: user.avatarUrl ?? "")) { image in
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } placeholder: {
                Color.gray
            }
            .frame(width: 50, height: 50)
            .clipShape(Circle())

            VStack(alignment: .leading, spacing: 4) {
                Text(user.displayName ?? user.username)
                    .font(.headline)

                if let bio = user.bio {
                    Text(bio)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }

            Spacer()

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 8)
    }
}
```

**Forms:**

```swift
// Views/Profile/SettingsView.swift
import SwiftUI

struct SettingsView: View {
    @State private var emailNotifications = true
    @State private var pushNotifications = true
    @State private var theme: Theme = .auto
    @State private var visibility: Visibility = .public

    enum Theme: String, CaseIterable, Identifiable {
        case light = "Light"
        case dark = "Dark"
        case auto = "Auto"

        var id: String { rawValue }
    }

    enum Visibility: String, CaseIterable, Identifiable {
        case `public` = "Public"
        case friends = "Friends"
        case `private` = "Private"

        var id: String { rawValue }
    }

    var body: some View {
        Form {
            Section("Notifications") {
                Toggle("Email Notifications", isOn: $emailNotifications)
                Toggle("Push Notifications", isOn: $pushNotifications)
            }

            Section("Appearance") {
                Picker("Theme", selection: $theme) {
                    ForEach(Theme.allCases) { theme in
                        Text(theme.rawValue).tag(theme)
                    }
                }
            }

            Section("Privacy") {
                Picker("Profile Visibility", selection: $visibility) {
                    ForEach(Visibility.allCases) { visibility in
                        Text(visibility.rawValue).tag(visibility)
                    }
                }
            }

            Section {
                Button("Sign Out", role: .destructive) {
                    // Sign out logic
                }
            }
        }
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.inline)
    }
}
```

### iOS Haptic Feedback

```swift
// Services/HapticService.swift
import UIKit

enum HapticService {
    static func impact(style: UIImpactFeedbackGenerator.FeedbackStyle) {
        let generator = UIImpactFeedbackGenerator(style: style)
        generator.impactOccurred()
    }

    static func notification(type: UINotificationFeedbackGenerator.FeedbackType) {
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(type)
    }

    static func selection() {
        let generator = UISelectionFeedbackGenerator()
        generator.selectionChanged()
    }
}

// Usage in SwiftUI
Button("Delete") {
    HapticService.notification(type: .warning)
    // Delete logic
}

Button("Like") {
    HapticService.impact(style: .medium)
    // Like logic
}
```

---

## Component Libraries

### Shared Component Specification

**Button Component Spec:**

```yaml
component: Button
variants:
  - primary (filled)
  - secondary (outlined)
  - tertiary (text)
  - destructive (error state)

sizes:
  - small (32px height)
  - medium (40px height)
  - large (48px height)

states:
  - default
  - hover
  - active (pressed)
  - focused
  - disabled
  - loading

props:
  - label: string (required)
  - icon: string (optional)
  - iconPosition: "left" | "right" (default: "left")
  - onClick: function (required)
  - disabled: boolean (default: false)
  - loading: boolean (default: false)
  - fullWidth: boolean (default: false)

accessibility:
  - role: button
  - aria-label: computed from label or explicit prop
  - aria-disabled: when disabled=true
  - aria-busy: when loading=true
  - keyboard: Enter/Space triggers onClick

platforms:
  web:
    tag: <button>
    focus: outline-offset: 2px
  ios:
    component: Button with ButtonStyle
    haptic: light impact on press
  android:
    component: @Composable Button
    ripple: true
```

### Web Component Library (React)

```typescript
// components/ui/Button.tsx
import React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-lg font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        primary: 'bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800',
        secondary: 'border-2 border-gray-300 bg-white text-gray-900 hover:bg-gray-50',
        tertiary: 'text-blue-600 hover:bg-blue-50 active:bg-blue-100',
        destructive: 'bg-red-600 text-white hover:bg-red-700 active:bg-red-800'
      },
      size: {
        sm: 'h-8 px-3 text-sm',
        md: 'h-10 px-4 text-base',
        lg: 'h-12 px-6 text-lg'
      },
      fullWidth: {
        true: 'w-full'
      }
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md'
    }
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  loading?: boolean;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant,
      size,
      fullWidth,
      loading,
      icon,
      iconPosition = 'left',
      children,
      disabled,
      ...props
    },
    ref
  ) => {
    return (
      <button
        ref={ref}
        className={cn(buttonVariants({ variant, size, fullWidth, className }))}
        disabled={disabled || loading}
        aria-busy={loading}
        {...props}
      >
        {loading && (
          <svg
            className="mr-2 h-4 w-4 animate-spin"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
        )}
        {!loading && icon && iconPosition === 'left' && (
          <span className="mr-2">{icon}</span>
        )}
        {children}
        {!loading && icon && iconPosition === 'right' && (
          <span className="ml-2">{icon}</span>
        )}
      </button>
    );
  }
);

Button.displayName = 'Button';
```

### iOS Component Library (SwiftUI)

```swift
// Components/Button.swift
import SwiftUI

enum ButtonVariant {
    case primary
    case secondary
    case tertiary
    case destructive
}

enum ButtonSize {
    case small
    case medium
    case large

    var height: CGFloat {
        switch self {
        case .small: return 32
        case .medium: return 40
        case .large: return 48
        }
    }

    var fontSize: CGFloat {
        switch self {
        case .small: return 14
        case .medium: return 16
        case .large: return 18
        }
    }
}

struct AppButton: View {
    let label: String
    let variant: ButtonVariant
    let size: ButtonSize
    let icon: String?
    let iconPosition: IconPosition
    let fullWidth: Bool
    let loading: Bool
    let action: () -> Void

    enum IconPosition {
        case left
        case right
    }

    init(
        _ label: String,
        variant: ButtonVariant = .primary,
        size: ButtonSize = .medium,
        icon: String? = nil,
        iconPosition: IconPosition = .left,
        fullWidth: Bool = false,
        loading: Bool = false,
        action: @escaping () -> Void
    ) {
        self.label = label
        self.variant = variant
        self.size = size
        self.icon = icon
        self.iconPosition = iconPosition
        self.fullWidth = fullWidth
        self.loading = loading
        self.action = action
    }

    var body: some View {
        Button(action: {
            HapticService.impact(style: .light)
            action()
        }) {
            HStack(spacing: 8) {
                if loading {
                    ProgressView()
                        .tint(textColor)
                } else {
                    if let icon = icon, iconPosition == .left {
                        Image(systemName: icon)
                    }

                    Text(label)
                        .font(.system(size: size.fontSize, weight: .medium))

                    if let icon = icon, iconPosition == .right {
                        Image(systemName: icon)
                    }
                }
            }
            .foregroundStyle(textColor)
            .frame(maxWidth: fullWidth ? .infinity : nil)
            .frame(height: size.height)
            .padding(.horizontal, horizontalPadding)
            .background(backgroundColor)
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(borderColor, lineWidth: variant == .secondary ? 2 : 0)
            )
        }
        .disabled(loading)
    }

    private var backgroundColor: Color {
        switch variant {
        case .primary: return .brandPrimary
        case .secondary: return .clear
        case .tertiary: return .clear
        case .destructive: return .red
        }
    }

    private var textColor: Color {
        switch variant {
        case .primary: return .white
        case .secondary: return .brandPrimary
        case .tertiary: return .brandPrimary
        case .destructive: return .white
        }
    }

    private var borderColor: Color {
        variant == .secondary ? .brandPrimary : .clear
    }

    private var horizontalPadding: CGFloat {
        switch size {
        case .small: return 12
        case .medium: return 16
        case .large: return 24
        }
    }
}
```

### Android Component Library (Jetpack Compose)

```kotlin
// ui/components/Button.kt
package com.example.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp

enum class AppButtonVariant {
    PRIMARY, SECONDARY, TERTIARY, DESTRUCTIVE
}

enum class AppButtonSize {
    SMALL, MEDIUM, LARGE
}

@Composable
fun AppButton(
    label: String,
    onClick: () -> Void,
    modifier: Modifier = Modifier,
    variant: AppButtonVariant = AppButtonVariant.PRIMARY,
    size: AppButtonSize = AppButtonSize.MEDIUM,
    icon: ImageVector? = null,
    iconPosition: IconPosition = IconPosition.LEFT,
    fullWidth: Boolean = false,
    loading: Boolean = false,
    enabled: Boolean = true
) {
    val height = when (size) {
        AppButtonSize.SMALL -> 32.dp
        AppButtonSize.MEDIUM -> 40.dp
        AppButtonSize.LARGE -> 48.dp
    }

    val horizontalPadding = when (size) {
        AppButtonSize.SMALL -> 12.dp
        AppButtonSize.MEDIUM -> 16.dp
        AppButtonSize.LARGE -> 24.dp
    }

    val buttonModifier = if (fullWidth) {
        modifier.fillMaxWidth().height(height)
    } else {
        modifier.height(height)
    }

    when (variant) {
        AppButtonVariant.PRIMARY -> {
            Button(
                onClick = onClick,
                modifier = buttonModifier,
                enabled = enabled && !loading,
                contentPadding = PaddingValues(horizontal = horizontalPadding)
            ) {
                ButtonContent(label, icon, iconPosition, loading)
            }
        }
        AppButtonVariant.SECONDARY -> {
            OutlinedButton(
                onClick = onClick,
                modifier = buttonModifier,
                enabled = enabled && !loading,
                contentPadding = PaddingValues(horizontal = horizontalPadding)
            ) {
                ButtonContent(label, icon, iconPosition, loading)
            }
        }
        AppButtonVariant.TERTIARY -> {
            TextButton(
                onClick = onClick,
                modifier = buttonModifier,
                enabled = enabled && !loading,
                contentPadding = PaddingValues(horizontal = horizontalPadding)
            ) {
                ButtonContent(label, icon, iconPosition, loading)
            }
        }
        AppButtonVariant.DESTRUCTIVE -> {
            Button(
                onClick = onClick,
                modifier = buttonModifier,
                enabled = enabled && !loading,
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.error,
                    contentColor = MaterialTheme.colorScheme.onError
                ),
                contentPadding = PaddingValues(horizontal = horizontalPadding)
            ) {
                ButtonContent(label, icon, iconPosition, loading)
            }
        }
    }
}

@Composable
private fun ButtonContent(
    label: String,
    icon: ImageVector?,
    iconPosition: IconPosition,
    loading: Boolean
) {
    if (loading) {
        CircularProgressIndicator(
            modifier = Modifier.size(20.dp),
            strokeWidth = 2.dp
        )
    } else {
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.wrapContentSize()
        ) {
            if (icon != null && iconPosition == IconPosition.LEFT) {
                Icon(icon, contentDescription = null, modifier = Modifier.size(20.dp))
            }

            Text(label)

            if (icon != null && iconPosition == IconPosition.RIGHT) {
                Icon(icon, contentDescription = null, modifier = Modifier.size(20.dp))
            }
        }
    }
}

enum class IconPosition {
    LEFT, RIGHT
}
```

---

## Accessibility Standards

### WCAG 2.2 Level AA Compliance

**Color Contrast Requirements:**

```
Normal Text (< 18pt or < 14pt bold):
  - AA: 4.5:1
  - AAA: 7:1

Large Text (>= 18pt or >= 14pt bold):
  - AA: 3:1
  - AAA: 4.5:1

UI Components and Graphics:
  - AA: 3:1
```

**Contrast Checker Tool:**

```typescript
// lib/utils/accessibility.ts
export function getContrastRatio(foreground: string, background: string): number {
  const l1 = getLuminance(foreground);
  const l2 = getLuminance(background);

  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);

  return (lighter + 0.05) / (darker + 0.05);
}

function getLuminance(hex: string): number {
  const rgb = hexToRgb(hex);
  const [r, g, b] = rgb.map(val => {
    val = val / 255;
    return val <= 0.03928 ? val / 12.92 : Math.pow((val + 0.055) / 1.055, 2.4);
  });

  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function meetsWCAG_AA(foreground: string, background: string, isLargeText: boolean = false): boolean {
  const ratio = getContrastRatio(foreground, background);
  return isLargeText ? ratio >= 3 : ratio >= 4.5;
}
```

### Platform-Specific Accessibility

**Web Accessibility (ARIA):**

```typescript
// components/ui/Dialog.tsx
export function Dialog({ title, children, onClose }: DialogProps) {
  return (
    <div
      role="dialog"
      aria-labelledby="dialog-title"
      aria-describedby="dialog-description"
      aria-modal="true"
    >
      <div className="dialog-overlay" onClick={onClose} aria-hidden="true" />

      <div className="dialog-content">
        <h2 id="dialog-title">{title}</h2>

        <button
          onClick={onClose}
          aria-label="Close dialog"
          className="close-button"
        >
          <CloseIcon aria-hidden="true" />
        </button>

        <div id="dialog-description">
          {children}
        </div>
      </div>
    </div>
  );
}
```

**iOS Accessibility:**

```swift
// Views/Components/AccessibleCard.swift
struct AccessibleCard: View {
    let title: String
    let description: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 8) {
                Text(title)
                    .font(.headline)
                Text(description)
                    .font(.body)
                    .foregroundStyle(.secondary)
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(.systemBackground))
            .cornerRadius(12)
        }
        .accessibilityLabel("\(title). \(description)")
        .accessibilityHint("Double tap to view details")
        .accessibilityAddTraits(.isButton)
    }
}
```

**Android Accessibility:**

```kotlin
@Composable
fun AccessibleCard(
    title: String,
    description: String,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {
                contentDescription = "$title. $description. Double tap to view details"
                role = Role.Button
            }
            .clickable(onClick = onClick)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium
            )
            Text(
                text = description,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
```

---

This design systems guide provides comprehensive patterns for building consistent, accessible, and platform-appropriate user interfaces across web, iOS, and Android platforms.
