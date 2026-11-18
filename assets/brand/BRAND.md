# Kera logo kit

The logo is the **K-bolt**: a K whose arm is a lightning bolt. Kera is short for *Keraunos*, Zeus's thunderbolt.

## Colours

| Name | Hex | Use |
|---|---|---|
| Bolt gold | `#FFC21A` | The K, on dark backgrounds only |
| Storm ink | `#0D1524` | Dark backgrounds, and the logo on light backgrounds |
| Light | `#F2F4F7` | Light backgrounds |
| White | `#FFFFFF` | The ERA letters on dark backgrounds |

## Font

The wordmark is set in **Unbounded ExtraBold (800)**, which is free to use, including commercially (SIL Open Font License). In the logo, the letters and the K-bolt have slightly rounded corners, so typing KERA in the font won't match it. The letters in the files are already shapes, so nobody needs the font installed. Always use the files.

## Files

| File | What it's for |
|---|---|
| `svg/kera-logo-on-dark.svg` | Gold K and white ERA, for dark backgrounds |
| `svg/kera-logo-on-light.svg` | All ink, for light backgrounds |
| `svg/kera-logo-white.svg` | All white, for photos or coloured backgrounds |
| `svg/kera-mark-gold.svg`, `-ink.svg`, `-white.svg` | The K-bolt on its own |
| `svg/kera-labs-logo-on-dark.svg`, `-on-light.svg` | The Kera Labs logo, for the organisation (Hugging Face, the GitHub organisation): KERA with small, spaced-out LABS beside it on the same baseline. Use plain KERA for the app itself |
| `social/banner-kera-labs-1200x300.png` | The Kera Labs banner for the top of the Hugging Face org card |
| `social/banner-1200x300.png` | The same banner with KERA only |
| `png/` | The same as images: logos 2000 px wide, the K 1024 px tall |
| `icons/app-icon.svg`, `icons/app-icon-1024.png` | Mac app icon artwork (Apple's 1024 px layout) |
| `icons/kera.icns` | Mac app icon file |
| `icons/kera.ico` | Windows app icon file (16 to 256 px) |
| `icons/favicon.svg`, `icons/favicon.ico`, `icons/apple-touch-icon.png` | Icons for the app's local web page |
| `social/social-preview-1280x640.png` | The picture GitHub shows when someone shares the repo link |

## Rules

- **Gold K only on dark backgrounds.** Gold on white is hard to see, so use the ink version on light backgrounds.
- **Keep clear space** around the logo at least as wide as the K's thick stem.
- **Smallest sizes:** the K on its own at 16 px; the full KERA logo at 24 px tall.
- **Don't** stretch it, tilt it, recolour parts of it, add shadows or glows, or put it on busy backgrounds.

## README header that switches with GitHub's light and dark mode

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/brand/png/kera-logo-on-dark-2000.png">
  <img alt="Kera" src="assets/brand/png/kera-logo-on-light-2000.png" width="360">
</picture>
```

## Small Mac icon sizes

`kera.icns` holds the 32 px size and up, and macOS shrinks it for the 16 px spots. For hand-tuned tiny sizes, rebuild it on a Mac from `icons/app-icon-1024.png` with Apple's `iconutil`.
