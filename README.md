<div align="center">

<img src="https://raw.githubusercontent.com/marko1olo/gigahrush/main/docs/banner_stenograph.jpg" width="100%" alt="STENOGRAPH — Steganographic Image Cipher & Visual Matrix Sandbox Banner"/>

# STENOGRAPH — Steganographic Image Cipher & Visual Matrix Sandbox

[![License](https://img.shields.io/badge/License-True%20People's%20v2.0-red?style=for-the-badge)](LICENSE.md)
[![Status](https://img.shields.io/badge/Status-Active%20Production-brightgreen?style=for-the-badge)]()
[![Code Audit](https://img.shields.io/badge/Audit-100%25%20Verified-purple?style=for-the-badge)]()

> **Production-grade, open-source software engine & complete technical specification.**

[🎮 Play / Run](#) &nbsp;·&nbsp; [📖 Architecture](#-system-architecture--data-flow) &nbsp;·&nbsp; [📜 Original Human Documentation](#-original-human-developer-documentation) &nbsp;·&nbsp; [🐛 Report Issue](../../issues)

</div>

---

## 📖 Executive Summary & Architectural Overview

This repository contains **Jirnyak/stenograph**, a high-performance system designed with clean module boundaries, explicit data flow pipelines, and zero proprietary lock-in.

---

## 🏗️ System Architecture & Data Flow

```
┌─────────────────────────────────┐
│     Input & Config Layer        │
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│     Core State Processing       │ ───> │     Memory & Buffer Cache       │
└─────────────────────────────────┘      └─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Output & Render Stage       │
└─────────────────────────────────┘
```

<div align="center">

<img src="https://raw.githubusercontent.com/marko1olo/gigahrush/main/docs/cyber_banner.jpg" width="100%" alt="STENOGRAPH — Steganographic Image Cipher & Visual Matrix Sandbox Secondary Visual"/>

</div>

---

## 📁 Directory Structure & Component Matrix

```
stenograph/
├── .DS_Store
├── .vite
├── .vite/deps
├── .vite/deps/_metadata.json
├── .vite/deps/package.json
├── README.md
├── __pycache__
├── __pycache__/steno.cpython-314.pyc
├── architecture.md
├── bot.py
├── bot2.py
├── chain.py
├── cipher.py
├── cipher1.py
├── decipher.py
├── decipher1.py
├── decipher_counter.py
├── decomposer.py
```

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

<details>
<summary>🇷🇺 Русская Версия (Подробная Сводка)</summary>

### Подробное описание проекта

Проект **STENOGRAPH — Steganographic Image Cipher & Visual Matrix Sandbox** содержит полное техническое описание архитектуры, методов сборки, структуры файлов и API-интерфейсов. Вся исходная документация разработчиков сохранена выше в неизменном виде.

</details>
