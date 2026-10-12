import Anthropic from '@anthropic-ai/sdk';
import { strictFormat } from '../utils/text.js';
import { getKey } from '../utils/keys.js';

// A prompt can be split with this marker for prompt caching: the text before it must not change between requests
// (e.g. persona, command docs) and is cached; the text after it changes every request (e.g. stats, inventory)
// and is moved to the end of the last message, so the conversation history can be cached too.
const DYNAMIC_MARKER = '<<<DYNAMIC>>>';

export class Claude {
    static prefix = 'anthropic';
    constructor(model_name, url, params) {
        this.model_name = model_name;
        this.params = params || {};

        let config = {};
        if (url)
            config.baseURL = url;
        
        config.apiKey = getKey('ANTHROPIC_API_KEY');

        this.anthropic = new Anthropic(config);
    }

    async sendRequest(turns, systemMessage) {
        const messages = strictFormat(turns);
        // newer models (4.6+) reject requests that end with an assistant message (prefill)
        if (messages[messages.length - 1].role === 'assistant') {
            messages.push({role: 'user', content: '_'});
        }
        let system = systemMessage;
        const marker_index = systemMessage?.indexOf(DYNAMIC_MARKER) ?? -1;
        if (marker_index !== -1) {
            const stable = systemMessage.slice(0, marker_index).trim();
            const dynamic = systemMessage.slice(marker_index + DYNAMIC_MARKER.length).trim();
            system = [{type: 'text', text: stable, cache_control: {type: 'ephemeral'}}];
            const last = messages[messages.length - 1];
            const content = typeof last.content === 'string'
                ? [{type: 'text', text: last.content || '_'}]
                : [...last.content];
            content[content.length - 1] = {...content[content.length - 1], cache_control: {type: 'ephemeral'}};
            if (dynamic)
                content.push({type: 'text', text: dynamic});
            last.content = content;
        }
        let res = null;
        try {
            console.log(`Awaiting anthropic response from ${this.model_name}...`)
            if (!this.params.max_tokens) {
                if (this.params.thinking?.budget_tokens) {
                    this.params.max_tokens = this.params.thinking.budget_tokens + 1000;
                    // max_tokens must be greater than thinking.budget_tokens
                } else {
                    // newer models think by default, and thinking tokens count toward max_tokens
                    this.params.max_tokens = 16000;
                }
            }
            const start = Date.now();
            const resp = await this.anthropic.messages.create({
                model: this.model_name || "claude-sonnet-4-6",
                system: system,
                messages: messages,
                ...(this.params || {})
            });

            const usage = resp.usage;
            console.log(`Received. (${Date.now() - start}ms, input: ${usage.input_tokens}, cache read: ${usage.cache_read_input_tokens ?? 0}, cache write: ${usage.cache_creation_input_tokens ?? 0}, output: ${usage.output_tokens})`);
            if (resp.stop_reason === 'refusal') {
                console.warn('Claude refused to respond:', resp.stop_details);
                return "Sorry, I can't respond to that.";
            }
            // get first content of type text
            const textContent = resp.content.find(content => content.type === 'text');
            if (textContent) {
                res = textContent.text;
            } else {
                console.warn(`No text content found in the response. (stop_reason: ${resp.stop_reason})`);
                res = 'No response from Claude.';
            }
        }
        catch (err) {
            if (err.message.includes("does not support image input")) {
                res = "Vision is only supported by certain models.";
            } else {
                res = "My brain disconnected, try again.";
            }
            console.log(err);
        }
        return res;
    }

    async sendVisionRequest(turns, systemMessage, imageBuffer) {
        const imageMessages = [...turns];
        imageMessages.push({
            role: "user",
            content: [
                {
                    type: "text",
                    text: systemMessage
                },
                {
                    type: "image",
                    source: {
                        type: "base64",
                        media_type: "image/jpeg",
                        data: imageBuffer.toString('base64')
                    }
                }
            ]
        });

        return this.sendRequest(imageMessages, systemMessage);
    }

    async embed(text) {
        throw new Error('Embeddings are not supported by Claude.');
    }
}
