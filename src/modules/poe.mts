const cleanPoeReasoning = (reasoning: string): string => {
    // Remove everything before the first line that starts with >
    const lines = reasoning.split('\n');
    let firstQuoteLineIndex = -1;

    // Find the first line that starts with >
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line && line.trim().startsWith('>')) {
            firstQuoteLineIndex = i;
            break;
        }
    }

    // If no quote line found, return empty
    if (firstQuoteLineIndex < 0) {
        return '';
    }

    // Take only the lines from the first quote line onwards
    const reasoningLines = lines.slice(firstQuoteLineIndex);

    // Remove ">" quotation markdown from the beginning of lines
    const cleanedLines = reasoningLines.map(line => {
        if (line.startsWith('> ')) {
            return line.substring(2); // Remove "> "
        } else if (line.startsWith('>')) {
            return line.substring(1); // Remove ">"
        }
        return line;
    });

    return cleanedLines.join('\n').trim();
}

export const startsWithThinking = (text: string): boolean => {
    // Remove leading whitespace and empty newlines
    const lines = text.split('\n');
    const nonEmptyLines: string[] = [];

    for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed !== '') {
            nonEmptyLines.push(trimmed);
        }
    }

    // Check if the first or second non-empty line starts with >
    if (nonEmptyLines.length >= 1 && nonEmptyLines[0] && nonEmptyLines[0].startsWith('>')) {
        return true;
    }
    if (nonEmptyLines.length >= 2 && nonEmptyLines[1] && nonEmptyLines[1].startsWith('>')) {
        return true;
    }

    return false;
}

export const findThinkingIndex = (text: string): number => {
    // Find where the actual reasoning content (>) starts
    // Look for the first line that starts with > and return the position of the >
    const lines = text.split('\n');
    let currentIndex = 0;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (!line) {
            currentIndex += 1; // Just the newline
            continue;
        }

        const trimmedLine = line.trim();

        // If this line starts with >, find the exact position of the >
        if (trimmedLine.startsWith('>')) {
            return currentIndex;
        }

        // Add the length of this line plus the newline character
        currentIndex += line.length + 1; // +1 for the \n
    }

    return -1; // No reasoning content found
}

export const cleanPoeReasoningDelta = (delta: string, isFirstDelta: boolean = false): string => {
    let cleanedDelta = delta;

    // For first delta, remove everything before the first line that starts with >
    if (isFirstDelta) {
        const lines = cleanedDelta.split('\n');
        let firstQuoteLineIndex = -1;

        // Find the first line that starts with >
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (line && line.trim().startsWith('>')) {
                firstQuoteLineIndex = i;
                break;
            }
        }

        // If we found a quote line, remove everything before it
        if (firstQuoteLineIndex >= 0) {
            cleanedDelta = lines.slice(firstQuoteLineIndex).join('\n');
        }
    }

    // Remove ">" quotation markdown from the beginning of lines
    const lines = cleanedDelta.split('\n');
    const cleanedLines = lines.map(line => {
        if (line.startsWith('> ')) {
            return line.substring(2); // Remove "> "
        } else if (line.startsWith('>')) {
            return line.substring(1); // Remove ">"
        }
        return line;
    });

    return cleanedLines.join('\n');
}

export const extractPoeReasoning = (text: string): { reasoning: string, content: string } => {
    if (!startsWithThinking(text)) {
        return { reasoning: '', content: text };
    }

    // Find where "Thinking..." starts
    const thinkingIndex = findThinkingIndex(text);
    if (thinkingIndex < 0) {
        return { reasoning: '', content: text };
    }

    const beforeThinking = text.substring(0, thinkingIndex);
    const afterThinking = text.substring(thinkingIndex);

    // Find the end of reasoning (two consecutive newlines where the second doesn't start with >)
    const lines = afterThinking.split('\n');
    let reasoningEndIndex = -1;

    for (let i = 0; i < lines.length - 1; i++) {
        const currentLine = lines[i];
        const nextLine = lines[i + 1];
        if (currentLine === '' && nextLine && nextLine !== '' && !nextLine.startsWith('>')) {
            // Found the end, calculate the position in the original text
            const linesBefore = lines.slice(0, i).join('\n');
            reasoningEndIndex = thinkingIndex + linesBefore.length + 1; // +1 for the newline
            break;
        }
    }

    if (reasoningEndIndex >= 0) {
        let reasoning = text.substring(thinkingIndex, reasoningEndIndex);
        const content = beforeThinking + text.substring(reasoningEndIndex + 1); // +1 to skip the newline

        // Clean up reasoning content
        reasoning = cleanPoeReasoning(reasoning);

        return { reasoning, content };
    } else {
        // No clear end found, treat everything after Thinking... as reasoning
        let reasoning = afterThinking;

        // Clean up reasoning content
        reasoning = cleanPoeReasoning(reasoning);

        return { reasoning, content: beforeThinking };
    }
}

export const modifyMessagesForPoe = (messages: any[], reasoning_effort?: string): any[] => {
    if (!reasoning_effort) return messages;

    // Find the last user message
    const modifiedMessages = [...messages];
    for (let i = modifiedMessages.length - 1; i >= 0; i--) {
        const message = modifiedMessages[i];
        if (message.role === 'user') {
            // Clone the message to avoid mutating the original
            const modifiedMessage = { ...message };

            // Handle different content types
            if (typeof modifiedMessage.content === 'string') {
                modifiedMessage.content = `${modifiedMessage.content} --reasoning_effort "${reasoning_effort}"`;
            } else if (Array.isArray(modifiedMessage.content)) {
                // Find the last text part and append to it
                const contentCopy = [...modifiedMessage.content];
                for (let j = contentCopy.length - 1; j >= 0; j--) {
                    const part = contentCopy[j];
                    if (part.type === 'text' && typeof part.text === 'string') {
                        contentCopy[j] = {
                            ...part,
                            text: `${part.text} --reasoning_effort "${reasoning_effort}"`
                        };
                        break;
                    }
                }
                modifiedMessage.content = contentCopy;
            }

            modifiedMessages[i] = modifiedMessage;
            break;
        }
    }

    return modifiedMessages;
}