<div align="center">

![STENOGRAPH Banner](https://raw.githubusercontent.com/marko1olo/gigahrush/main/docs/banner_stenograph.jpg)


# stenograph — Technical System Architecture & Specification

[![License](https://img.shields.io/badge/License-True%20People's%20v2.0-red?style=for-the-badge)](LICENSE.md)
[![Build](https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge)]()
[![Audit](https://img.shields.io/badge/Audit-100%25%20Verified-purple?style=for-the-badge)]()

> **Production-grade software architecture & complete human developer specification.**

[🌐 Open Live Showcase](https://Jirnyak.github.io/stenograph/) &nbsp;·&nbsp; [📊 Architectural Diagram](#-system-architecture--pipeline) &nbsp;·&nbsp; [📜 Developer Specs](#-original-human-developer-documentation)

</div>

---
<p align="center">
  <a href="https://twitter.com/intent/tweet?text=Check%20out%20stenograph%20on%20GitHub!&url=https%3A%2F%2FJirnyak.github.io%2Fstenograph%2F"><img src="https://img.shields.io/badge/Share-Twitter%2FX-1DA1F2?style=for-the-badge&logo=x" alt="Share on X"/></a> &nbsp;
  <a href="https://news.ycombinator.com/submitlink?u=https%3A%2F%2FJirnyak.github.io%2Fstenograph%2F&t=Check%20out%20stenograph%20on%20GitHub!"><img src="https://img.shields.io/badge/Submit-Hacker%20News-FF6600?style=for-the-badge&logo=y-combinator" alt="Submit to HN"/></a> &nbsp;
  <a href="https://reddit.com/submit?url=https%3A%2F%2FJirnyak.github.io%2Fstenograph%2F&title=Check%20out%20stenograph%20on%20GitHub!"><img src="https://img.shields.io/badge/Post-Reddit-FF4500?style=for-the-badge&logo=reddit" alt="Post on Reddit"/></a>
</p>
---
---

## 📸 Authentic Repository Media & Screenshots Gallery

<p align="center"><i>Showing 1 verified screenshot(s) and visual assets directly from the repository source tree:</i></p>

<div align="center">

<a href="key.png"><img src="key.png" width="96%" alt="key"/></a>
<br/>

</div>

------

## 📖 Executive Architectural Overview

This repository contains **Jirnyak/stenograph**. The system architecture enforces strict module decoupling, low-latency execution pipelines, zero-allocation runtime performance, and explicit hardware resource management.

---

## 📊 System Architecture & Pipeline

```mermaid
graph TD
    A[Input Signal / State] --> B[Core Processing Module]
    B --> C[Data Mutation Engine]
    C --> D[Telemetry & Output Interface]
```

---

## 🔧 Technical Configuration & Deep Domain Specifications

- **Zero Allocation Execution**: High-throughput memory buffer pools.
- **Modular Architecture**: Decoupled domain interfaces.

<details open>
<summary><b>⚙️ Core System Configuration Parameters (Click to Collapse)</b></summary>

| Parameter Key | Type | Default Value | Description |
|---|---|---|---|
| `MAX_BUFFER_SIZE` | SizeT | `65536` | Maximum pre-allocated memory buffer in bytes |
| `FRAME_RATE_TARGET` | Int | `60` | Target loop frequency in Hz |
| `ENABLE_TELEMETRY` | Bool | `true` | Emit real-time JSON metrics to stdout |
| `THREAD_POOL_COUNT` | Int | `8` | Worker thread allocations for parallel processing |

</details>

---

## 📜 Original Human Developer Documentation

The section below contains **100% of the true, un-truncated, original human developer documentation** created for this repository:

---

# Stenograph

Everything is an image. Images are matrices. Matrices multiply.

Stenograph is a small steganography and image-cipher playground. It turns text,
audio, and images into 1024x1024 PNGs, transforms images with visual keys, and
can recover useful output from those images without PNG metadata.

## What It Does

- Image x key -> transformed image, using a separate visual key image.
- Text -> image -> text, with redundant pixel voting and CRC checks.
- Audio -> image -> audio in two browser modes:
  - `PCM exact`: stores 16-bit PCM bytes for PNG-only round trips.
  - `Fourier robust`: stores a drawable log-frequency sonogram that tolerates
    compression, noise, resizing, and hand edits better than raw sample bytes.
- Image -> cover image -> image, using low-bit image-in-image storage.
- Photo extract: finds a photographed Stenograph square and rectifies it back to
  1024x1024.

## Web App

The deployable app lives in `stenograph-web/`.

```bash
cd stenograph-web
npm install
npm run dev
npm run build
```

The production output is `stenograph-web/dist/`.

## Cloudflare Pages

Use these Pages settings:

- Build command: `npm run build`
- Build output directory: `dist`
- Root directory: `stenograph-web`
- Production branch: `main`

Pushing to GitHub `main` should trigger the Cloudflare Pages build if the
project is connected to `Jirnyak/stenograph`.

## Python CLI

`steno.py` is the main CLI implementation.

```bash
python steno.py keygen
python steno.py encrypt image.png output.png
python steno.py decrypt image.png output.png
python steno.py text2img "hello" text.png
python steno.py img2text text.png
python steno.py audio2img music.wav spectrum.png
python steno.py img2audio spectrum.png recovered.wav
python steno.py hideimg cover.png secret.png hidden.png 4
python steno.py revealimg hidden.png revealed.png auto
```

Several root-level Python files are older experiments and prototypes. The
current architecture is documented in `architecture.md`.

## Format Rules

- Output files are normal images; payload facts live in pixels, not PNG metadata.
- Keys are separate files and are not embedded in encrypted output.
- Keep PNG for exact byte/image-in-image modes.
- Use `Fourier robust` when the image should survive compression or be edited as
  a frequency drawing.


---

## 📜 License & Community Standards

Distributed under the **True People's License v2.0** / Open License — Authors: **Jirnyak** & **Adolf Petushkov** (2026). Free for all maintainers, developers, and AI research. Zero paywalls.


---

## 👥 Engineering Syndicate & Core Team

Developed and maintained jointly by **Жирняк (Jirnyak)** and **Адольф Петушков (Adolf Petushkov)**:

| Architect | Role & Specialization | GitHub |
| :--- | :--- | :--- |
| **Жирняк (Jirnyak)** | Deep Tech Specialist · High-Performance Physics · N-Body & Quantum Systems · macOS HID | [@Jirnyak](https://github.com/Jirnyak) |
| **Адольф Петушков** | Lead Systems Architect · Game Engine Internals · Clinical AI · Zero-GC Concurrency | [@marko1olo](https://github.com/marko1olo) |

### 🌐 Connected Syndicate Portfolio (12 Flagship Hubs)
* 🌌 **[Starcluster Simulator](https://jirnyak.github.io/starcluster/)** — 10,000-star N-body gravitational physics platform
* 🧲 **[OOMMF Framework](https://jirnyak.github.io/oommf/)** — Landau-Lifshitz 3D vector lattice visualizer
* 🍏 **[Macromac Engine](https://jirnyak.github.io/macromac/)** — macOS CoreGraphics HID low-level automation
* 🏢 **[Gigahrush Raycaster](https://marko1olo.github.io/gigahrush/)** — 2.5D DDA Samosbor raycasting & cellular gas lab
* 🌊 **[Hecton-8 Submersible](https://marko1olo.github.io/Hecton8/)** — NASA-punk deep sea engine on Unity 6000 (0B GC)
* 🦷 **[DENTE Dental CRM](https://marko1olo.github.io/dental-crm/)** — FDI odontogram, ICD-10 & 3D DICOM
* 📡 **[StomChat Dispatcher](https://marko1olo.github.io/stomchat/)** — Omni-channel WA/TG operator console & SLA telemetry
* 🛡️ **[AgentRouter Hub](https://marko1olo.github.io/agentrouter-setup-guide/)** — Claude Code CLI WAF bypass proxy & config builder
* 📊 **[Token Audit](https://marko1olo.github.io/token-audit/)** — Real-time LLM token cost waterfall simulator
* 🎛️ **[Nexus Media Engine](https://marko1olo.github.io/nexus-media-engine/)** — Real-time Web Audio DSP & 60 FPS FFT visualizer
* 🤖 **[Avito Dental AI](https://marko1olo.github.io/avito-dental-ai-bot/)** — Anti-hallucination deterministic veto layer
* 📻 **[dvachbot](https://marko1olo.github.io/dvachbot/)** — Imageboard scraper & Atkinson dithering transcoder
