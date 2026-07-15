// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Replika",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
    ],
    targets: [
        .target(name: "ReplikaCore"),
        .target(
            name: "SpeechSwiftProvider",
            dependencies: [
                "ReplikaCore",
                .product(name: "Qwen3ASR", package: "speech-swift"),
                .product(name: "SpeechVAD", package: "speech-swift")
            ]
        ),
        .executableTarget(
            name: "asr-probe",
            dependencies: [.product(name: "Qwen3ASR", package: "speech-swift")]
        ),
        .executableTarget(
            name: "replika-spike",
            dependencies: ["ReplikaCore", "SpeechSwiftProvider"]
        ),
        .testTarget(name: "ReplikaCoreTests", dependencies: ["ReplikaCore"]),
        .testTarget(
            name: "SpeechSwiftProviderTests",
            dependencies: ["SpeechSwiftProvider", "ReplikaCore"]
        )
    ]
)
