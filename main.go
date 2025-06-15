package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// Config holds the application's configuration.
type Config struct {
	APIKey        string
	InitialPrompt string
}

// loadConfig reads the API key and initial prompt from their respective files.
func loadConfig() (*Config, error) {
	// Read the API key from key.txt
	keyBytes, err := os.ReadFile("key.txt")
	if err != nil {
		return nil, fmt.Errorf("failed to read API key from key.txt: %w. Please ensure the file exists", err)
	}
	apiKey := strings.TrimSpace(string(keyBytes))
	if apiKey == "" {
		return nil, errors.New("API key file 'key.txt' is empty")
	}

	// Read the initial personality prompt from personality.jb
	// This part is optional, so we handle a "not found" error gracefully.
	promptBytes, err := os.ReadFile("personality.jb")
	var initialPrompt string
	if err != nil {
		if os.IsNotExist(err) {
			log.Println("No 'personality.jb' file found, starting a standard chat session.")
		} else {
			// For any other error, we should probably know about it.
			log.Printf("Warning: could not read personality.jb: %v", err)
		}
	} else {
		initialPrompt = string(promptBytes)
	}

	return &Config{
		APIKey:        apiKey,
		InitialPrompt: initialPrompt,
	}, nil
}

// runChatSession handles the main interactive loop with the user.
func runChatSession(cs *genai.ChatSession, ctx context.Context) {
	fmt.Println("Your conversational AI is ready. Type 'quit' to exit.")
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				log.Printf("Error reading input: %v", err)
			}
			break
		}
		userInput := scanner.Text()

		if strings.ToLower(userInput) == "quit" {
			fmt.Println("Goodbye!")
			break
		}

		if userInput == "" {
			continue
		}

		fmt.Print("Gemini: ...") // Provide instant feedback
		resp, err := cs.SendMessage(ctx, genai.Text(userInput))
		if err != nil {
			fmt.Print("\r") // Clear the "Gemini: ..." line
			log.Printf("Error sending message: %v", err)
			continue
		}
		fmt.Print("\r") // Clear the "Gemini: ..." line
		printResponse(resp)
	}
}

// printResponse iterates through the model's response and prints the text.
func printResponse(resp *genai.GenerateContentResponse) {
	fmt.Print("Gemini: ")
	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				if txt, ok := part.(genai.Text); ok {
					fmt.Print(txt)
				}
			}
		}
	}
	fmt.Println() // Add a newline for better formatting
}

func main() {
	// --- 1. Load Configuration ---
	cfg, err := loadConfig()
	if err != nil {
		log.Fatalf("Initialization failed: %v", err)
	}

	// The context will manage the lifecycle of our API requests.
	ctx := context.Background()

	// --- 2. Initialize the AI Client ---
	client, err := genai.NewClient(ctx, option.WithAPIKey(cfg.APIKey))
	if err != nil {
		log.Fatalf("Failed to create AI client: %v", err)
	}
	defer client.Close()

	model := client.GenerativeModel("gemini-2.0-flash")

	// --- 3. Start a Chat Session ---
	cs := model.StartChat()

	// If an initial prompt was loaded, send it to the model first to set the personality.
	if cfg.InitialPrompt != "" {
		log.Println("Sending initial personality prompt...")
		_, err := cs.SendMessage(ctx, genai.Text(cfg.InitialPrompt))
		if err != nil {
			log.Fatalf("Failed to send initial prompt: %v", err)
		}
	}

	// --- 4. Run the main application loop ---
	runChatSession(cs, ctx)
}