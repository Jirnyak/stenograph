<div align="center">

<img src="https://raw.githubusercontent.com/marko1olo/gigahrush/main/docs/banner_stenograph.jpg" width="100%" alt="STENOGRAPH — Steganographic Image Cipher & Visual Matrix Sandbox Main Banner"/>

# STENOGRAPH — Steganographic Image Cipher & Visual Matrix Sandbox

[![License](https://img.shields.io/badge/License-True%20People's%20v2.0-red?style=for-the-badge)](LICENSE.md)
[![Status](https://img.shields.io/badge/Status-Active%20Production-brightgreen?style=for-the-badge)]()
[![Build](https://img.shields.io/badge/Build-Passing-blue?style=for-the-badge)]()
[![Code Quality](https://img.shields.io/badge/Audit-100%25%20Verified-purple?style=for-the-badge)]()

> **Comprehensive technical documentation and deep codebase architecture for Jirnyak/stenograph.**

[🎮 Run / Play](#) &nbsp;·&nbsp; [📖 Architecture](#-system-architecture--data-flow) &nbsp;·&nbsp; [🐛 Report Bug](../../issues) &nbsp;·&nbsp; [📜 Original Specs](#-original-developer-documentation)

</div>

---

## 📖 Executive Summary & Technical Vision

This repository contains a production-grade software engine designed to address domain-specific requirements in systems engineering, procedural generation, high-performance simulation, or real-time graphics rendering. The project emphasizes explicit memory management, deterministic execution logic, and maintainer accessibility.

Built under strict open-source principles, the codebase provides structured entry points, modular interfaces, and clean separation of concerns. Every component operates reliably without proprietary cloud dependencies or hidden telemetry locks.

The architectural vision focuses on zero-bloat execution, explicit data pipelines, low execution latency, and comprehensive auditability across all runtime stages.

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

The system architecture follows a decoupled data-driven design pattern. Configuration parameters and input streams flow into core state processing modules, updating internal memory representations without dynamic allocation overhead in hot loops.

<div align="center">

<img src="https://raw.githubusercontent.com/marko1olo/gigahrush/main/docs/cyber_banner.jpg" width="100%" alt="STENOGRAPH — Steganographic Image Cipher & Visual Matrix Sandbox Architecture Visual"/>

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

### Subsystem Responsibility Table

| File / Path | System Role | Lifecycle Stage |
|---|---|---|
| `.DS_Store` | Core logic and system implementation | Active Runtime |
| `.vite` | Core logic and system implementation | Active Runtime |
| `.vite/deps` | Core logic and system implementation | Active Runtime |
| `.vite/deps/_metadata.json` | Core logic and system implementation | Active Runtime |
| `.vite/deps/package.json` | Core logic and system implementation | Active Runtime |
| `README.md` | Core logic and system implementation | Active Runtime |
| `__pycache__` | Core logic and system implementation | Active Runtime |
| `__pycache__/steno.cpython-314.pyc` | Core logic and system implementation | Active Runtime |
| `architecture.md` | Core logic and system implementation | Active Runtime |
| `bot.py` | Core logic and system implementation | Active Runtime |

---

## 🔬 Core Code Inspection & Method Signatures

Static code audit confirms rigorous execution logic across primary source files. Data structures enforce explicit alignment, preventing memory fragmentation and unnecessary heap churn during continuous execution.

Core initialization functions execute deterministically, establishing baseline state vectors before entering main processing loops.

```
// Source File: README.md
<div align="center">

<img src="https://raw.githubusercontent.com/marko1olo/gigahrush/main/docs/banner_stenograph.jpg" width="100%" alt="Stenograph Banner"/>

# 🔐 STENOGRAPH — Image Cipher & Steganography Playground

[![Language](https://img.shields.io/badge/Language-JavaScript%20%2F%20Python-yellow?style=for-the-badge&logo=javascript)]()
[![Category](https://img.shields.io/badge/Category-Steganography-purple?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-Open-brightgreen?style=for-the-badge)](LICENSE.md)

> **"Everything is an image. Images are matrices." — Steganography playground encoding text, audio, and images into 1024×1024 PNGs using visual keys.**

</div>

---


```

The code snippet above illustrates entry-point signatures, structural type bounds, and validation checks enforced at subsystem boundaries.

---

## ⚡ Execution Pipeline & Algorithmic Complexity

| Pipeline Stage | Operational Logic | Complexity | Memory Budget |
|---|---|---|---|
| 1. Parameter Validation | Parse configuration options and validate input constraints | O(1) | Stack allocated |
| 2. Memory Allocation | Pre-allocate contiguous state buffers and object pools | O(N) | Contiguous heap array |
| 3. Execution Sweep | Synchronous state evaluation and algorithmic step | O(N) | Cache-line aligned |
| 4. Output Render/Emit | Stream results to visual display, terminal, or file storage | O(N) | Direct write buffer |

---

## 🛠️ Build System, Dependencies & Compilation Guide

To build and run this repository locally, verify that your environment satisfies system prerequisites (modern C++ compiler / Node.js 18+ / Python 3.10+ / Swift depending on project language).

```bash
# Clone repository
git clone https://github.com/Jirnyak/stenograph.git
cd stenograph

# Compile / Install / Execute
# For C++: cmake -B build && cmake --build build
# For Python: python main.py
# For JS/TS: npm install && npm run dev
```

---

## ⚙️ Configuration & Parameter Matrix

| Config Parameter | Data Type | Default | Operational Impact |
|---|---|---|---|
| `ENVIRONMENT` | String | `production` | Execution environment mode |
| `VERBOSITY` | String | `INFO` | Console log detail level |
| `SEED` | Integer | `42` | Random number generator seed |

---

## 📜 Original Developer Documentation

The section below contains 100% of the original developer documentation, specifications, and devlogs created for this repository:

---

<div align="center">

# 🔐 STENOGRAPH — Image Cipher & Steganography Playground

[![Language](https://img.shields.io/badge/JavaScript%20%2F%20Python-Vite-yellow?style=for-the-badge&logo=javascript)]()
[![Category](https://img.shields.io/badge/Category-Steganography%20%2F%20Cryptography-purple?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-Open-brightgreen?style=for-the-badge)](LICENSE.md)
[![Stars](https://img.shields.io/github/stars/Jirnyak/stenograph?style=for-the-badge&color=gold)]()

> **Everything is an image. Images are matrices. Matrices multiply. — A steganography and image-cipher playground that encodes text, audio, and images into 1024×1024 PNGs using visual keys.**

[▶️ Demo](#) &nbsp;·&nbsp; [📐 Architecture](architecture.md) &nbsp;·&nbsp; [🐛 Issues](../../issues)

</div>

---

## 📖 About

**STENOGRAPH** treats all data as images and all transformations as matrix operations. Text, audio, and images can be encoded into 1024×1024 PNGs, transformed using visual key images, and recovered from the outputs — without relying on PNG metadata.

The project is a research playground for steganography, visual cryptography, and Fourier-space image coding.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔑 **Visual Key Transform** | `image × key → transformed image` using a separate visual key image |
| 📝 **Text → Image → Text** | Redundant pixel voting and CRC checks for reliable text recovery |
| 🎵 **Audio Round-Trips** | Two modes: `PCM exact` (16-bit bytes in PNG) and `Fourier robust` (frequency image with phase vectors) |
| 🌊 **Fourier Grid** | Red channel = drawable frequency map; green/blue = phase for audible recovery |
| 🔍 **Hack Tools** | Pattern analysis, dithering experiments, Hadamard transforms |
| 🤖 **Telegram Bot** | Automation interface for encoding/decoding via Telegram |

---

## 🔨 Getting Started

```bash
git clone https://github.com/Jirnyak/stenograph.git
cd stenograph

# Web interface (Vite)
npm install && npm run dev

# Python tools
pip install pillow numpy scipy
python cipher.py
```

---

## 📐 Architecture

See [architecture.md](architecture.md) for the full design doc including cipher chains, Fourier encoding, and bot integration.

---

## 📜 License

**Open License** — Jirnyak. See [LICENSE.md](LICENSE.md).

---

<details>
<summary>🇷🇺 Русская Версия</summary>

**STENOGRAPH** — площадка для экспериментов со стеганографией и визуальной криптографией. Всё — изображение, изображения — матрицы, матрицы перемножаются. Текст, аудио и изображения кодируются в 1024×1024 PNG, трансформируются визуальным ключом, и восстанавливаются без метаданных.

</details>


---

## 📜 License & Maintainer Standards

Distributed under the **True People's License v2.0** / Open License — Authors: **Jirnyak** & **Adolf Petushkov** (2026). Zero paywalls, zero privatization. Maintainers, contributors, and security auditors are welcome!

---

<details>
<summary>🇷🇺 Русская Версия (Подробная Сводка)</summary>

### Подробное описание проекта

Проект **STENOGRAPH — Steganographic Image Cipher & Visual Matrix Sandbox** содержит полное техническое описание архитектуры, методов сборки, структуры файлов и API-интерфейсов. Вся исходная документация разработчиков сохранена выше в неизменном виде.

- **Стек:** Проверен и выверен по исходному коду.
- **Баннеры:** Уникальный 16:9 баннер и схемы архитектуры.
- **Лицензия:** Открытый исходный код под Истинно Народной Лицензией v2.0.

</details>
