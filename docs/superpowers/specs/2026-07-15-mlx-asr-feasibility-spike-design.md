# Дизайн: Feasibility-спайк локального MLX-ASR (Replika)

- **Дата:** 2026-07-15
- **Статус:** согласован, готов к плану реализации
- **Тип:** feasibility-спайк (первый срез проекта)
- **Родительский документ:** [`architecture.md`](../../../architecture.md)

---

## 1. Контекст и цель

Replika — нативное macOS-приложение для транскрипции с диаризацией на Apple Silicon
(Qwen3-ASR через MLX, диаризация на Neural Engine). Проект greenfield.

Прежде чем вкладываться в полный стек, снимаем **главный технический риск**: реально ли
завести Qwen3-ASR со словными таймкодами и диаризацией нативно на этой машине.

**Одна фраза цели:** доказать, что файловая транскрипция Qwen3-ASR со словными таймкодами
и метками спикеров работает нативно через `speech-swift` за нашим протоколом
`TranscriptionProvider`, с приемлемой скоростью/памятью/качеством.

### Что показала разведка экосистемы

- **MLX-порт Qwen3-ASR существует** и не один: квантизированные веса уже лежат на HF
  (`mlx-community/Qwen3-ASR-1.7B` и `0.6B`, 4/5/8-bit) — конвертация не нужна.
- **`soniqo/speech-swift`** (Apache-2.0, SwiftPM, модульные таргеты) покрывает наш стек
  почти 1:1: Qwen3-ASR, Qwen3-ForcedAligner, Silero VAD, Sortformer-диаризация, стриминг.
- Подстраховка: `Blaizzy/mlx-audio-swift` (Swift SDK на MLX) и `moona3k/mlx-qwen3-asr`
  (чистая Python-референс-реализация, Apache-2.0) — если решим портировать сами.

### Стратегическое решение

**Обёртка над готовым** (`speech-swift`) за нашим протоколом `TranscriptionProvider`.
Быстрее всего даёт ответ «да/нет» и одновременно ветит зависимость. Граница-протокол
делает выбор **обратимым**: позже сможем vendor'ить/форкать/заменить без изменения UI.

---

## 2. Границы

### В scope

- Файловый режим: аудиофайл → транскрипт со словными таймкодами.
- **Диаризация** (Sortformer) + мёрж меток спикеров в сегменты — полный цикл
  «текст + метки спикеров», т.к. это и есть «нужный формат».
- Обёртка за минимальным `TranscriptionProvider` (async-stream).
- Замеры: RTF, peak unified memory, время загрузки — для 4-bit и 8-bit.
- CLI-харнесс (`swift run`).

### Вне scope (YAGNI для спайка)

- Живой/стриминг режим, VAD-чанкинг длинного аудио.
- Менеджер моделей / загрузчик / реестр / манифесты (за нас работает `fromPretrained`).
- Именование спикеров (тривиальный rename-маппинг — заглушка) и voiceprint-энролмент.
- Remote-провайдер, персистентность, библиотека, поиск, экспорт.
- Полноценный SwiftUI (следующий срез — после подтверждения спайка).

---

## 3. Критерии приёмки (pass/fail)

1. **Заводится:** минимальный нативный macOS-таргет грузит Qwen3-ASR-1.7B (4-bit) через
   `speech-swift` и транскрибирует реальный файл в текст — офлайн, без Python, без подпроцессов.
2. **Словные таймкоды:** ForcedAligner отдаёт per-word start/end; таймкоды монотонны и
   лежат в пределах длительности аудио.
3. **Диаризация + мёрж:** Sortformer даёт ≥1 `SpeakerTag`; сегменты корректно размечены
   спикерами по таймкодам («Speaker 0: … / Speaker 1: …»).
4. **Граница-протокол реальна:** всё лежит за минимальным `TranscriptionProvider`
   (async-stream `TranscriptEvent`).
5. **Замерено:** RTF, peak unified memory, load time — для 4-bit и 8-bit, на RU- и
   мультиспикер-клипе — сведены в таблицу.
6. **Санити качества:** на слух транскрипт RU читается корректно; спот-чек, что word-ts
   совпадают со звуком.

---

## 4. Архитектура спайка

### Поток данных (файловый режим)

```
file
 └─ AudioLoader ─────────────► [Float] @ 16 kHz mono
      ├─ Sortformer ─────────► [SpeakerTag]  (кто когда)
      └─ Qwen3ASR.transcribe ► text
             └─ ForcedAligner.align ► [Word]  (start/end по слову)
                    └─ SegmentBuilder ► [Segment]  (группировка по паузам)
                           └─ SpeakerMerger ► [Segment + speaker]  (мёрж по таймкодам)
                                  └─ AsyncStream<TranscriptEvent> ► CLI-печать + Bench
```

### Домен-типы и протокол (файловый минимум из §1 architecture.md)

```swift
protocol TranscriptionProvider {
    var capabilities: ProviderCaps { get }
    func transcribe(_ audio: AudioSource,
                    options: TranscribeOptions) -> AsyncThrowingStream<TranscriptEvent, Error>
    func cancel()
}

enum TranscriptEvent {
    case progress(Double)
    case committed(Segment)     // финализированный сегмент
    case speaker(SpeakerTag)    // метка диаризации (мёржится по таймкодам)
    case done(Transcript)
}

struct Word     { let text: String; let start: Double; let end: Double }
struct Segment  { let text: String; let start: Double; let end: Double
                  let words: [Word]; var speaker: Int? }
struct SpeakerTag { let speaker: Int; let start: Double; let end: Double }
```

`TranscribeOptions` — immutable `struct` (frozen-аналог): язык (`auto|ru|en`), quant
(`q4|q8`), флаг диаризации, контекст-подсказка.

### Референс API `speech-swift`

```swift
import Qwen3ASR
let model = try await Qwen3ASRModel.fromPretrained()
let text  = model.transcribe(audio: samples, sampleRate: 16000)

let aligner = try await Qwen3ForcedAligner.fromPretrained()
let aligned = aligner.align(audio: samples, text: text, sampleRate: 24000)  // [Word]
```

### Структура файлов (принцип малых файлов, 200–400 строк)

```
Package.swift                         # SwiftPM: exe-таргет + speech-swift dep + тест-таргет
Sources/
  ReplikaCore/                        # домен (переиспользуется в продукте)
    TranscriptionProvider.swift       # протокол + ProviderCaps + TranscribeOptions
    TranscriptTypes.swift             # Segment, Word, SpeakerTag, Transcript, TranscriptEvent
    AudioLoader.swift                 # AVFoundation → 16k mono [Float]
    SegmentBuilder.swift              # words → segments (группировка по паузам)
    SpeakerMerger.swift               # мёрж speaker-меток в сегменты по таймкодам
  SpeechSwiftProvider/                # обёртка над зависимостью
    SpeechSwiftProvider.swift         # conforms to TranscriptionProvider
    ProviderRegistry.swift            # factory/registry (перенос духа правил на Swift)
  replika-spike/                      # CLI
    main.swift                        # путь к файлу + флаги --quant/--diarize
    Bench.swift                       # RTF, peak RAM (mach task_info), load time, таблица
Tests/
  ReplikaCoreTests/
    SpikeSmokeTests.swift             # .done непустой; word-ts монотонны; ≥1 speaker
```

---

## 5. Тестовые данные и замеры

- **Аудио:** собственные записи пользователя — RU и мультиспикер (встреча/звонок). Самый
  честный тест. Численный WER не считаем (нет эталона); санити — на слух по известному
  контенту. Если позже нужен численный WER — добавим один публичный клип с эталоном.
- **Bench-таблица:** строки — `quant × clip`, колонки — `RTF | peak RAM | load time | WER-санити`.
- **RTF** = длительность аудио / время стены; **peak RAM** — через `mach task_info`.

---

## 6. Тестирование и стиль

- **Smoke-тест** (swift-testing) на коротком встроенном клипе: провайдер эмитит `.done` с
  непустым текстом; word-ts монотонны и в пределах длительности; при диаризации — ≥1
  `SpeakerTag`. Регрессионный якорь без переусложнения.
- **Стиль (дух правил проекта, перенос на Swift):** файлы 200–400 строк; `struct`/
  immutability; протокол + реестр провайдеров; `os.Logger` вместо `print`; конфиг структурой;
  Swift 6 strict concurrency. (mypy/ruff/pytest неприменимы.)

---

## 7. Ограничения и известные зазоры

- **Min deployment target: macOS 15+**, Swift 6, Xcode 16 + Metal Toolchain, Apple Silicon.
  Причина — `MLState` (резидентный KV-кэш на ANE) в `speech-swift`. Зафиксировано.
- **Кэш моделей:** `speech-swift` тянет веса в свой `~/Library/Caches/qwen3-speech/`, а не в
  наш `Application Support` из §5. Для спайка ок; для продукта — отдельный срез (перенос
  хранилища через `cacheDir:`/`QWEN3_CACHE_DIR`, реестр, манифесты).
- **Сегменты:** `transcribe` отдаёт plain text; сегменты собираем из word-ts алайнера
  (`SegmentBuilder`, группировка по паузам) — деталь, которую спайк валидирует на практике.

---

## 8. Риски, которые спайк вскроет

- Реальное качество RU у `speech-swift`.
- Собирается ли как SwiftPM-CLI (Swift 6 / Metal Toolchain / MLState) без app-бандла и
  entitlements.
- Работает ли Sortformer/ANE из CLI без app-контекста.
- Насколько корректно `SegmentBuilder` строит сегменты из одних word-ts.
- Перф/память vs ожидание (сравнение 4-bit и 8-bit).

---

## 9. Definition of Done

- CLI транскрибирует реальный RU/мультиспикер файл → сегменты со словными таймкодами и
  метками спикеров, офлайн.
- Всё за `TranscriptionProvider`; провайдер сменяем через реестр.
- Bench-таблица (4-bit/8-bit × клипы) заполнена и зафиксирована в репо.
- Smoke-тест зелёный.
- Короткий вывод: подтверждён ли путь «обёртка над speech-swift» для продукта, какие
  зазоры вскрылись, что берём в следующий срез.

---

## 10. Следующие срезы (после спайка)

Порядок из §7 architecture.md, уточнённый находками:

1. **SwiftUI-форма** «нужного формата»: лента с сегментами, метки спикеров, rename-маппинг
   (именование), плеер/скраббер. Первое, что хочет увидеть пользователь после спайка.
2. **Менеджер моделей**: перенос хранилища в `Application Support`, реестр, манифесты, докачка.
3. **Живой режим**: стриминг Qwen3-ASR + streaming Sortformer, системный звук (ScreenCaptureKit).
4. **Remote-провайдер** (OpenAI-совместимый) как альтернативная реализация протокола.
5. **Инспектор** (обработка через `TextProvider`) + семантический поиск, экспорт.
6. **Voiceprint-энролмент** (узнавание голоса между записями).
