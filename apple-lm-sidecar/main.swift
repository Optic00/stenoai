import Foundation
import FoundationModels

@main
struct StenoAppleLM {
    static func main() async {
        let command = CommandLine.arguments.dropFirst().first ?? "status"
        do {
            switch command {
            case "status":
                try printStatus()
            case "complete":
                try await printComplete()
            case "stream":
                try await printStream()
            default:
                FileHandle.standardError.write(Data("usage: steno-apple-lm status | complete | stream\n".utf8))
                exit(2)
            }
        } catch {
            emitError(error)
            exit(1)
        }
    }
}

private func readStdinPrompt() -> String {
    let data = FileHandle.standardInput.readDataToEndOfFile()
    return String(data: data, encoding: .utf8) ?? ""
}

private func printJSON(_ object: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: object, options: [])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

private func emitError(_ error: Error) {
    // Fixed keys only — never echo the prompt or model output.
    var reason = "generation_failed"
#if compiler(>=6.4)
    if #available(macOS 27.0, *) {
        if let modelError = error as? LanguageModelError {
            reason = languageModelErrorReason(modelError)
        } else if let modelError = error as? SystemLanguageModel.Error {
            reason = systemLanguageModelErrorReason(modelError)
        } else if let sessionError = error as? LanguageModelSession.Error {
            reason = languageModelSessionErrorReason(sessionError)
        } else if let generationError = error as? LanguageModelSession.GenerationError {
            reason = generationErrorReason(generationError)
        }
    } else if #available(macOS 26.0, *),
              let generationError = error as? LanguageModelSession.GenerationError {
        reason = generationErrorReason(generationError)
    }
#else
    if #available(macOS 26.0, *),
       let generationError = error as? LanguageModelSession.GenerationError {
        reason = generationErrorReason(generationError)
    }
#endif
    try? printJSON(["error": "apple_lm_failed", "reason": reason])
}

#if compiler(>=6.4)
@available(macOS 27.0, *)
private func languageModelErrorReason(_ error: LanguageModelError) -> String {
    switch error {
    case .contextSizeExceeded:
        return "context_window"
    case .rateLimited:
        return "rate_limited"
    case .guardrailViolation:
        return "guardrail"
    case .refusal:
        return "refusal"
    case .unsupportedLanguageOrLocale:
        return "unsupported_language"
    case .timeout:
        return "timeout"
    case .unsupportedCapability,
         .unsupportedTranscriptContent,
         .unsupportedGenerationGuide:
        return "generation_failed"
    @unknown default:
        return "generation_failed"
    }
}

@available(macOS 27.0, *)
private func systemLanguageModelErrorReason(
    _ error: SystemLanguageModel.Error
) -> String {
    switch error {
    case .assetsUnavailable:
        return "assets_unavailable"
    @unknown default:
        return "generation_failed"
    }
}

@available(macOS 27.0, *)
private func languageModelSessionErrorReason(
    _ error: LanguageModelSession.Error
) -> String {
    switch error {
    case .concurrentRequests:
        return "concurrent_requests"
    case .transcriptMutationWhileResponding:
        return "generation_failed"
    @unknown default:
        return "generation_failed"
    }
}
#endif

@available(macOS 26.0, *)
private func generationErrorReason(
    _ error: LanguageModelSession.GenerationError
) -> String {
    switch error {
    case .exceededContextWindowSize:
        return "context_window"
    case .assetsUnavailable:
        return "assets_unavailable"
    case .guardrailViolation:
        return "guardrail"
    case .unsupportedGuide, .decodingFailure:
        return "generation_failed"
    case .unsupportedLanguageOrLocale:
        return "unsupported_language"
    case .rateLimited:
        return "rate_limited"
    case .concurrentRequests:
        return "concurrent_requests"
    case .refusal:
        return "refusal"
    @unknown default:
        return "generation_failed"
    }
}

private func printStatus() throws {
    guard #available(macOS 26.0, *) else {
        try printJSON(["available": false, "reason": "unsupported_os"])
        return
    }
    let model = SystemLanguageModel.default
    switch model.availability {
    case .available:
        // Build against the macOS 26 SDK used by the release runner. Variant
        // inspection is a macOS 27 SDK API, and #available cannot make that
        // symbol compile against an older SDK. The OS still chooses the model.
        try printJSON([
            "available": true,
            "display_name": "Apple Intelligence",
        ])
    case .unavailable(let reason):
        try printJSON([
            "available": false,
            "reason": unavailableReasonName(reason),
        ])
    }
}

@available(macOS 26.0, *)
private func unavailableReasonName(
    _ reason: SystemLanguageModel.Availability.UnavailableReason
) -> String {
    switch reason {
    case .deviceNotEligible:
        return "deviceNotEligible"
    case .appleIntelligenceNotEnabled:
        return "appleIntelligenceNotEnabled"
    case .modelNotReady:
        return "modelNotReady"
    @unknown default:
        return "unavailable"
    }
}

private func printComplete() async throws {
    guard #available(macOS 26.0, *) else {
        try printJSON(["error": "apple_lm_failed", "reason": "unsupported_os"])
        exit(1)
    }
    let model = SystemLanguageModel.default
    guard model.isAvailable else {
        try printJSON(["error": "apple_lm_failed", "reason": "unavailable"])
        exit(1)
    }
    let session = LanguageModelSession(model: model)
    let response = try await session.respond(to: readStdinPrompt())
    try printJSON(["text": response.content])
}

private func printStream() async throws {
    guard #available(macOS 26.0, *) else {
        try printJSON(["error": "apple_lm_failed", "reason": "unsupported_os"])
        exit(1)
    }
    let model = SystemLanguageModel.default
    guard model.isAvailable else {
        try printJSON(["error": "apple_lm_failed", "reason": "unavailable"])
        exit(1)
    }
    let session = LanguageModelSession(model: model)
    var previous = ""
    let stream = session.streamResponse(to: readStdinPrompt())
    for try await snapshot in stream {
        let current = stringContent(snapshot.content)
        let delta: String
        if current.hasPrefix(previous) {
            delta = String(current.dropFirst(previous.count))
        } else {
            delta = current
        }
        previous = current
        if !delta.isEmpty {
            try printJSON(["delta": delta])
        }
    }
    try printJSON(["done": true])
}

private func stringContent(_ value: some Any) -> String {
    if let text = value as? String {
        return text
    }
    return String(describing: value)
}
