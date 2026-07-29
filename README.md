<div align="center">

<img src="https://raw.githubusercontent.com/marko1olo/gigahrush/main/docs/banner_stenograph.jpg" width="100%" alt="Stenograph Banner"/>

# 🔐 STENOGRAPH — Image Cipher & Steganography Playground

[![Language](https://img.shields.io/badge/Language-JavaScript%20%2F%20Python-yellow?style=for-the-badge&logo=javascript)]()
[![Category](https://img.shields.io/badge/Category-Steganography-purple?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-Open-brightgreen?style=for-the-badge)](LICENSE.md)

> **"Everything is an image. Images are matrices." — Steganography playground encoding text, audio, and images into 1024×1024 PNGs using visual keys.**

</div>

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
