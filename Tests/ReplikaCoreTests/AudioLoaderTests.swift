import Testing
import Foundation
import AVFoundation
@testable import ReplikaCore

private func writeSineWav(_ url: URL, seconds: Double, sampleRate: Double) throws {
    let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
                            channels: 1, interleaved: false)!
    let file = try AVAudioFile(forWriting: url, settings: fmt.settings)
    let frames = AVAudioFrameCount(seconds * sampleRate)
    let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: frames)!
    buf.frameLength = frames
    let ptr = buf.floatChannelData![0]
    for i in 0..<Int(frames) {
        ptr[i] = Float(sin(2.0 * .pi * 440.0 * Double(i) / sampleRate)) * 0.5
    }
    try file.write(from: buf)
}

@Test func loadsAndResamplesToMono16k() throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("sine_\(UUID().uuidString).wav")
    try writeSineWav(url, seconds: 1.0, sampleRate: 44100)
    defer { try? FileManager.default.removeItem(at: url) }

    let samples = try AudioLoader.loadMono16k(url)
    // ~16000 samples for 1 s, allow small converter slack
    #expect(abs(samples.count - 16000) < 400)
    #expect(samples.contains { $0 != 0 })
}
