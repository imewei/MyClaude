# Accessibility Standards Reference

Comprehensive guide to WCAG 2.1/2.2 compliance and accessible development.

## WCAG 2.1 Principles (POUR)

1. **Perceivable**: Information must be presentable to users in ways they can perceive
2. **Operable**: UI components must be operable
3. **Understandable**: Information and operation must be understandable
4. **Robust**: Content must be robust enough to work with assistive technologies

## Conformance Levels

- **Level A**: Minimum level (most basic accessibility features)
- **Level AA**: Target level for most organizations
- **Level AAA**: Highest level (not always achievable for all content)

**Recommendation**: Target WCAG 2.1 Level AA compliance.

---

## Key Success Criteria (WCAG 2.1 Level AA)

### 1.4.3 Contrast (Minimum) - AA

**Normal text**: 4.5:1 contrast ratio
**Large text** (18pt or 14pt bold): 3:1 contrast ratio

**Tools**:
- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
- Chrome DevTools Lighthouse audit

**Example**:
```css
/* ❌ Bad - Insufficient contrast */
color: #777; /* on white background = 4.47:1 */
background: #fff;

/* ✅ Good - Sufficient contrast */
color: #595959; /* on white background = 7:1 */
background: #fff;
```

### 1.1.1 Non-text Content - A

All images, icons, and non-text content must have text alternatives.

```html
<!-- ❌ Bad -->
<img src="chart.png">

<!-- ✅ Good -->
<img src="chart.png" alt="Bar chart showing 50% increase in sales">

<!-- Decorative images -->
<img src="decoration.png" alt="" role="presentation">
```

### 2.1.1 Keyboard - A

All functionality must be operable via keyboard.

```javascript
// ✅ Good - Keyboard accessible custom dropdown
<div
  role="button"
  tabindex="0"
  onKeyPress={(e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      toggleDropdown();
    }
  }}
  onClick={toggleDropdown}
>
  Open Menu
</div>
```

### 2.4.3 Focus Order - A

Focus order must be logical and intuitive.

```html
<!-- ✅ Good - Use tabindex sparingly -->
<button tabindex="0">Normal button</button>
<input type="text" tabindex="0">

<!-- ❌ Bad - Don't manipulate tab order unnecessarily -->
<button tabindex="5">Button</button>
```

### 2.4.7 Focus Visible - AA

Keyboard focus must be visible.

```css
/* ❌ Bad - Removing focus outline */
button:focus {
  outline: none;
}

/* ✅ Good - Custom focus indicator */
button:focus {
  outline: 2px solid #0066cc;
  outline-offset: 2px;
}

/* ✅ Better - Skip on mouse, show on keyboard */
button:focus:not(:focus-visible) {
  outline: none;
}
button:focus-visible {
  outline: 2px solid #0066cc;
}
```

### 3.2.2 On Input - A

Changing settings must not cause unexpected context changes.

```javascript
// ❌ Bad - Form submits on change
<select onChange={submitForm}>

// ✅ Good - Explicit submission
<select onChange={updatePreview}>
<button onClick={submitForm}>Submit</button>
```

### 4.1.2 Name, Role, Value - A

UI components must have accessible names and roles.

```html
<!-- ❌ Bad -->
<div onclick="handleClick()">Submit</div>

<!-- ✅ Good -->
<button type="submit">Submit</button>

<!-- ✅ Good - Custom component with ARIA -->
<div
  role="button"
  aria-label="Submit form"
  tabindex="0"
  onclick="handleClick()"
>
  Submit
</div>
```

---

## ARIA (Accessible Rich Internet Applications)

### ARIA Roles

**Landmark roles**:
```html
<header role="banner">
<nav role="navigation">
<main role="main">
<aside role="complementary">
<footer role="contentinfo">
```

**Widget roles**:
```html
<div role="dialog" aria-labelledby="dialog-title">
<div role="alertdialog" aria-describedby="alert-desc">
<div role="tablist">
  <button role="tab" aria-selected="true">Tab 1</button>
</div>
```

### ARIA States and Properties

**aria-label**: Provides accessible name
```html
<button aria-label="Close dialog">
  <svg><!-- X icon --></svg>
</button>
```

**aria-labelledby**: References another element for label
```html
<div role="dialog" aria-labelledby="dialog-title">
  <h2 id="dialog-title">Confirm Action</h2>
</div>
```

**aria-describedby**: Provides additional description
```html
<input
  type="password"
  aria-describedby="password-requirements"
>
<div id="password-requirements">
  Must be at least 8 characters
</div>
```

**aria-hidden**: Hides from screen readers
```html
<span aria-hidden="true">★★★★★</span>
<span class="sr-only">5 out of 5 stars</span>
```

**aria-expanded**: Indicates expanded/collapsed state
```html
<button aria-expanded="false" aria-controls="menu">
  Menu
</button>
<ul id="menu" hidden>...</ul>
```

**aria-live**: Announces dynamic content
```html
<div role="status" aria-live="polite">
  Item added to cart
</div>

<div role="alert" aria-live="assertive">
  Error: Please fill required fields
</div>
```

---

## Semantic HTML

Use semantic HTML elements instead of divs:

```html
<!-- ❌ Bad -->
<div class="header">
  <div class="nav">
    <div class="link">Home</div>
  </div>
</div>
<div class="main-content">
  <div class="article">...</div>
</div>

<!-- ✅ Good -->
<header>
  <nav>
    <a href="/">Home</a>
  </nav>
</header>
<main>
  <article>...</article>
</main>
```

---

## Form Accessibility

### Labels

```html
<!-- ❌ Bad - No label -->
<input type="text" placeholder="Email">

<!-- ✅ Good - Explicit label -->
<label for="email">Email</label>
<input type="text" id="email">

<!-- ✅ Also good - Implicit label -->
<label>
  Email
  <input type="text">
</label>
```

### Required Fields

```html
<label for="name">
  Name <span aria-label="required">*</span>
</label>
<input
  type="text"
  id="name"
  required
  aria-required="true"
>
```

### Error Messages

```html
<label for="email">Email</label>
<input
  type="email"
  id="email"
  aria-invalid="true"
  aria-describedby="email-error"
>
<div id="email-error" role="alert">
  Please enter a valid email address
</div>
```

### Fieldsets

```html
<fieldset>
  <legend>Shipping Address</legend>
  <label for="street">Street</label>
  <input type="text" id="street">
  <label for="city">City</label>
  <input type="text" id="city">
</fieldset>
```

---

## Skip Links

Allow keyboard users to skip repetitive content:

```html
<a href="#main-content" class="skip-link">
  Skip to main content
</a>
...
<main id="main-content">
  <!-- Main content -->
</main>
```

```css
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  background: #000;
  color: #fff;
  padding: 8px;
  z-index: 100;
}

.skip-link:focus {
  top: 0;
}
```

---

## Screen Reader Only Text

```css
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}
```

```html
<button>
  <svg><!-- Icon --></svg>
  <span class="sr-only">Delete item</span>
</button>
```

---

## Testing for Accessibility

### Automated Testing

**axe-core (JavaScript)**:
```javascript
const { AxePuppeteer } = require('@axe-core/puppeteer');
const puppeteer = require('puppeteer');

const browser = await puppeteer.launch();
const page = await browser.newPage();
await page.goto('https://example.com');

const results = await new AxePuppeteer(page).analyze();
console.log(results.violations);
```

**pa11y (CLI)**:
```bash
pa11y https://example.com --standard WCAG2AA --reporter cli
```

**Lighthouse**:
```bash
lighthouse https://example.com --only-categories=accessibility
```

### Manual Testing

**Keyboard Navigation**:
1. Navigate using Tab (forward) and Shift+Tab (backward)
2. Activate using Enter or Space
3. Close modals with Escape
4. All interactive elements must be reachable and operable

**Screen Reader Testing**:
- **NVDA** (Windows, free)
- **JAWS** (Windows, paid)
- **VoiceOver** (macOS/iOS, built-in)
- **TalkBack** (Android, built-in)

**Checklist**:
- [ ] All content is announced
- [ ] Navigation is logical
- [ ] Forms are properly labeled
- [ ] Error messages are clear
- [ ] Dynamic content changes are announced

---

## Common Patterns

### Modal Dialog

```html
<div
  role="dialog"
  aria-labelledby="dialog-title"
  aria-modal="true"
>
  <h2 id="dialog-title">Confirm Delete</h2>
  <p>Are you sure you want to delete this item?</p>
  <button>Cancel</button>
  <button>Delete</button>
</div>
```

```javascript
// Focus management
const modal = document.querySelector('[role="dialog"]');
const firstFocusable = modal.querySelector('button');
const lastFocusable = modal.querySelectorAll('button')[1];

// Trap focus within modal
modal.addEventListener('keydown', (e) => {
  if (e.key === 'Tab') {
    if (e.shiftKey && document.activeElement === firstFocusable) {
      e.preventDefault();
      lastFocusable.focus();
    } else if (!e.shiftKey && document.activeElement === lastFocusable) {
      e.preventDefault();
      firstFocusable.focus();
    }
  }
});
```

### Accordion

```html
<div class="accordion">
  <h3>
    <button
      aria-expanded="false"
      aria-controls="section1"
      id="accordion1"
    >
      Section 1
    </button>
  </h3>
  <div
    id="section1"
    role="region"
    aria-labelledby="accordion1"
    hidden
  >
    Content for section 1
  </div>
</div>
```

### Data Tables

```html
<table>
  <caption>Sales Report Q4 2024</caption>
  <thead>
    <tr>
      <th scope="col">Product</th>
      <th scope="col">Units Sold</th>
      <th scope="col">Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Widget A</th>
      <td>1,250</td>
      <td>$25,000</td>
    </tr>
  </tbody>
</table>
```

---

## Accessibility Checklist

### Every Release

- [ ] Automated accessibility tests pass
- [ ] Keyboard navigation works throughout
- [ ] Focus indicators are visible
- [ ] Color contrast meets WCAG AA
- [ ] All images have alt text
- [ ] Forms have proper labels
- [ ] Error messages are clear and associated
- [ ] Dynamic content changes are announced
- [ ] Semantic HTML used appropriately
- [ ] ARIA used only when necessary
- [ ] Tested with screen reader

---

## References

- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [MDN Accessibility](https://developer.mozilla.org/en-US/docs/Web/Accessibility)
- [A11y Project Checklist](https://www.a11yproject.com/checklist/)
- [WebAIM](https://webaim.org/)
