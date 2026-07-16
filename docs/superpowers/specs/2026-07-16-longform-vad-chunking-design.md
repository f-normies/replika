# Дизайн: VAD-chunked транскрипция длинного аудио (срез B)

- **Дата:** 2026-07-16
- **Статус:** дизайн утверждён, готов к плану
- **Тип:** design spec следующего среза
- **Родительские документы:** [`architecture.md`](../../../architecture.md),
  [`2026-07-15-mlx-asr-feasibility-spike-FINDINGS.md`](2026-07-15-mlx-asr-feasibility-spike-FINDINGS.md)
- **Ветка:** `spike/mlx-asr-feasibility` (продолжаем на ней)

---

## 1. Контекст и проблема

Feasibility-спайк подтвердил стратегию «обёртка над `speech-swift`» для клипов
≤ 60 с, но вскрыл **блокер** (FINDINGS §5): единственный вызов
`Qwen3ASRModel.transcribe(audio: весь_файл)` не масштабируется на длинное
аудио. На реальном 9.4-минутном звонке конвейер вернул **0 сегментов** —
жёсткий `maxTokens = 448` на весь вызов, нет оконного разбиения; пустой
ASR-выход каскадит в пустой align → пустые сегменты.

Этот срез снимает блокер: заменяем единственный `transcribe(весь файл)` на
**VAD-нарезку + per-span transcribe/align**, чтобы типичный звонок в минуты
давал корректные, размеченные по спикерам, таймштампованные сегменты.
Downstream-звенья (`SegmentBuilder`, `SpeakerMerger`, диаризация) не меняются —
они уже длины-агностичны (работают с плоским `[Word]` / `[SpeakerTag]`).

Планка среза: **product-quality** (строгие тесты, bench, обработка ошибок).

---

## 2. Сверка с реальным API `speech-swift` (запиненная ревизия)

Проверено чтением исходников запиненного чек-аута (`Package.resolved`), не по
памяти. Ключевые факты, на которые опирается дизайн:

- **Готовый VAD есть.** Модуль `SpeechVAD` отдаёт
  `SileroVADModel.detectSpeech(audio:sampleRate:) -> [SpeechSegment]`
  (`defaultModelId = "aufklarer/Silero-VAD-v6.2.1-MLX"`, `chunkSize = 512`).
  Свой VAD не пишем — переиспользуем этот.
- **`SpeechSegment { startTime: Float, endTime: Float }`** (секунды),
  `duration: Float`. `AlignedWord { text: String, startTime: Float, endTime:
  Float }`. Обе величины `Float` → конверсия в `Double` на границе с нашими
  доменными типами (как уже делает провайдер).
- **`VADConfig`** имеет пресеты `.sileroDefault` (onset 0.5 / offset 0.35 /
  minSpeechDuration 0.25 / minSilenceDuration 0.1) и `.default` (pyannote).
  Берём `.sileroDefault` как базу.
- **VAD-спаны не ограничены по длине сверху.** Непрерывная речь без пауз
  (монолог) может дать один спан длиннее нашего лимита. Поэтому поверх VAD
  нужен **force-split** — ровно как это делает апстримный `StreamingASR`
  через `maxSegmentDuration` (дефолт 10 с).
- **`align()` надёжен только до ~270 с** (комментарий в `ForcedAligner.swift`):
  дальше «плато» — все хвостовые слова получают один таймкод. `alignLong(...)`
  существует, чтобы это обходить внутренним чанкингом. **В нашем дизайне
  align вызывается per-span (≤ maxChunkSeconds ≈ 10 с), далеко под 270 с →
  `alignLong` не нужен**, обычный `align` безопасен.
- **Эталон апстрима.** `StreamingASR.transcribeStream(audio:)` уже делает
  Silero-VAD + per-span `transcribe` и отдаёт `TranscriptionSegment`. Мы **не**
  оборачиваем его: он не зовёт `ForcedAligner` (нет word-timestamps) и отдаёт
  сегменты по VAD-границам вместо нашей pause-based сегментации. Мы зеркалим
  его VAD+force-split-логику, но сохраняем наш пайплайн `words → SegmentBuilder
  → SpeakerMerger`.

---

## 3. Scope

**Спина (обязательно):**
- VAD-нарезка + per-span transcribe/align вместо single-shot.
- Чистая логика планирования чанков и сшивки слов (unit-тестируема без моделей).
- Провайдер-оркестрация + `ChunkConfig`.
- Unit-тесты + E2E на длинном клипе; расширение bench.

**Включаем (дёшево и напрямую задевается переписыванием align-звена):**
- Пломбинг реального языка в `align()` (перестаёт быть хардкод `"English"`).
- Hardening `AudioLoader` error-surface (отложенные Task-4 minors: unwrapped
  `file.read`, Int64→UInt32 truncation trap; error-path тесты).
- Перенос кэша моделей из `~/Library/Caches/qwen3-speech/` в
  `Application Support` через `cacheDir:` (FINDINGS зазор #2).

**Явно откладываем (за scope):**
- Полноценный model-manager / registry / докачка UX (FINDINGS #3).
- Live/streaming режим (FINDINGS #4).
- Автоматизация MLX `mlx.metallib` packaging для дистрибуции (FINDINGS #7).
- Отдельный observability-путь «диаризация без ASR» (FINDINGS §5, мелкий).

---

## 4. Архитектура и границы компонентов

Принцип — отделить **чистую логику** (тест без моделей) от **модель-зависимой
оркестрации** (E2E):

**`ReplikaCore` (чистое, без зависимости от `speech-swift`):**

- `ChunkPlanner` — из VAD-спанов `[(start, end)]` (сек) + `ChunkConfig` строит
  список ограниченных окон `[ChunkWindow]`: спаны длиннее `maxChunkSeconds`
  режутся force-split на под-окна с `overlapSeconds` перекрытием; короткие
  проходят как есть. Детерминированно, без I/O.
- `WordStitcher` — сшивает per-chunk слова (уже сдвинутые в абсолютное время)
  в единый монотонный `[Word]`; дедупит слова в overlap-регионах force-split.

**`SpeechSwiftProvider` (модель-зависимое):**

- Грузит `SileroVADModel` → `detectSpeech(samples)` → `[SpeechSegment]`.
- `ChunkPlanner.plan(spans, config)` → `[ChunkWindow]`.
- Цикл по окнам (последовательно): вырезает `spanAudio`, зовёт
  `asr.transcribe(spanAudio, language)`, затем `aligner.align(spanAudio, text,
  language)`, сдвигает локальные слова на `window.start`, копит.
- `WordStitcher.stitch(...)` → глобальные `words` → **существующие**
  `SegmentBuilder.build` + `SpeakerMerger.merge`.

Это сохраняет `capabilities.wordTimestamps: true` честным, доменные типы и
pause-based сегментацию — без изменений.

---

## 5. Поток данных

```
file → AudioLoader.loadMono16k → samples:[Float] (16 kHz)
  ├─ (diarize) SortformerDiarizer.diarize(samples) → tags:[SpeakerTag]   // на весь файл, как сейчас
  └─ VAD: SileroVADModel.detectSpeech(samples) → speechSpans:[SpeechSegment]
        → ChunkPlanner.plan(spans, config) → windows:[ChunkWindow]
        → for window in windows:                                        // последовательно
              try Task.checkCancellation()
              spanAudio = samples[window]
              text  = asr.transcribe(spanAudio, language)               // ≤ maxChunk сек → под 448-token cap
              local = aligner.align(spanAudio, text, language)          // ≤ maxChunk сек → под ~270с plateau
              words += local.map { offset by window.start }
        → WordStitcher.stitch(words, windows, overlap) → globalWords:[Word]
  → SegmentBuilder.build(globalWords) → base:[Segment]
  → SpeakerMerger.merge(base, tags) → merged:[Segment]
  → events: .speaker* , .progress(per-chunk) , .committed* , .done
```

---

## 6. Конфигурация чанкинга

Новый frozen-тип в `ReplikaCore`, пробрасывается через `TranscribeOptions`:

```swift
public struct ChunkConfig: Sendable, Equatable {
    public let maxChunkSeconds: Double   // дефолт 10.0 (совпадает с апстрим StreamingASR)
    public let overlapSeconds: Double    // дефолт 0.5 — применяется только на force-split
    public let vadTier: VadTier          // .silero (дефолт)
    public let minSpeechSeconds: Double  // прокидывается в VADConfig.minSpeechDuration
}
```

`maxChunkSeconds` консервативен: реплики в разговоре обычно < 10 с, force-split
ловит длинные монологи. Дефолты дают рабочий из коробки продукт.

---

## 7. Сшивка слов и overlap

- Между **разными VAD-спанами** overlap не нужен — они разделены тишиной,
  слова не рвутся. Простая конкатенация со сдвигом на `window.start`.
- Overlap важен только на **force-split** одного длинного непрерывного спана
  (тишины нет, риск разрезать слово). Берём небольшой `overlapSeconds` (0.5 с);
  `WordStitcher` дедупит: из пары окон в overlap-регионе оставляет слова одного
  окна по правилу «midpoint слова < граница окна». Гарантия монотонности
  `start` на выходе.
- **Fallback** (если overlap-дедуп окажется хрупким на реальном аудио):
  force-split без overlap, принимаем редкий разрез слова на стыке монолога.
  Задокументированный запасной вариант, не дефолт.

---

## 8. Обработка ошибок и прогресс

- **Ноль речевых спанов** (тишина/музыка) → пустой `Transcript` (`.done` с
  `segments: []`), не крэш.
- **Пустой `transcribe` на спане** → спан пропускается (trim → skip empty,
  как в апстриме).
- **Отмена** → `Task.checkCancellation()` в начале каждой итерации цикла.
  Побочный плюс: per-chunk даёт естественные точки отмены (сейчас отмена не
  прерывает синхронный `transcribe` на весь файл).
- **Прогресс** → заменяем фиксированные 0.1/0.4/0.7 на инкремент по чанкам:
  диаризация → `0.3`, затем `0.3 + 0.6 * (i / N)` по мере чанков, `.done` →
  `1.0`.
- **AudioLoader hardening** → `file.read` в do/catch с доменной ошибкой; guard
  на Int64→UInt32 переполнение длины на гигантских файлах; error-path тесты.

---

## 9. Тестирование (product-quality)

**Unit (без моделей, `ReplikaCoreTests`):**
- `ChunkPlanner`: пустой вход; один короткий спан; спан ровно `== maxChunk`;
  спан чуть больше (force-split на 2); очень длинный спан (несколько под-окон,
  корректный overlap); несколько спанов подряд.
- `WordStitcher`: сдвиг таймкодов; монотонность; дедуп в overlap; пустые чанки
  в середине.
- `AudioLoader`: error-path (битый / пустой файл).

**E2E (gated env, `SpeechSwiftProviderTests`):**
- Полный 9.4-мин клип (`SMOKE_AUDIO`): ассертим **> 0 сегментов**, **2
  спикера**, монотонные word-таймкоды, покрытие ≈ полной длительности. Это
  регресс-тест ровно на найденный блокер (FINDINGS §5).

Приватность (BINDING, из `progress.md`): аудио никогда не коммитится; тесты
читают путь из env (`SMOKE_AUDIO`), не из бандла; ноль транскрипт-текста в
логах/ассертах.

---

## 10. Bench

Расширяем `replika-spike` bench: строки длинного клипа теперь дают реальные
сегменты. Снимаем RTF / peak-RAM с чанкингом. Ожидание: peak RAM остаётся
ограниченным (спаны обрабатываются последовательно, per-span буферы малы;
+небольшой Silero-VAD в стек). Обновляем FINDINGS-таблицу: закрываем строку
«0 сегментов» реальными числами.

---

## 11. Риски и verify-шаги (для плана)

- **Диаризация на длинном буфере.** `SortformerDiarizer.diarize` вызывается на
  всём 9.4-мин аудио одним махом. FINDINGS §5 говорят «технически может
  отработать», но это никогда не surfaced. **Verify:** подтвердить, что
  диаризатор не деградирует на такой длине (адекватное число спикеров, разумные
  границы). Если деградирует — чанкуем и его (есть `SortformerStreamingState` /
  `StreamingVADProcessor`). Это отдельный риск, помечен явно.
- **Overlap-дедуп на реальной речи.** Правило «midpoint < граница» проверить на
  E2E; при хрупкости — fallback §7.
- **Peak RAM с добавленным VAD.** Замерить в bench; Silero мал, но фиксируем.

---

## 12. Утверждённые решения (резюме)

1. Подход **B** (свой VAD-chunk слой поверх нашего пайплайна), не A (обёртка
   `StreamingASR`) и не C (фиксированные окна + LCP).
2. Align **per-span**, `alignLong` не используется.
3. VAD — Silero (`.sileroDefault`), переиспользуем апстрим.
4. `maxChunkSeconds = 10`, `overlapSeconds = 0.5` (overlap только на force-split).
5. Планка — product-quality; включаем aligner-language + AudioLoader-hardening +
   cache-relocation; остальное откладываем.
