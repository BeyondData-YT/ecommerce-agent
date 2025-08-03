# E-commerce AI Agent

This project aims to build a robust, real-world e-commerce agent from the ground up, starting with core functionalities and progressively adding advanced features. This repository is a companion to our YouTube series, where we will guide you through the process of creating a production-ready, deployable product.

The idea behind this agent is to provide instant, accurate, and personalized support to customers, freeing up human agents for more complex issues. It can be used to answer product questions, provide recommendations, assist with orders, and much more. By focusing on a layered architecture and modern best practices, this project is not just about the final product; it's about teaching you how to build robust, scalable, and maintainable AI applications.

## Project Goal

The primary objective is to create a fully functional, multi-modal AI agent capable of assisting customers with their queries. We will start with a basic text-based agent and progressively add more advanced functionalities. The core of our agent will be a Retrieval-Augmented Generation (RAG) system that uses a PostgreSQL vector database to provide accurate, context-aware responses.

## What We're Building

The idea behind this agent is to address all customer needs through a single, intelligent interface. We will build an agent that can:

* **Understand Natural Language:** Go beyond simple keywords to truly comprehend user intent.
* **Answer Product Questions:** Provide instant and accurate responses on product details, stock availability, and recommendations based on customer preferences.
* **Understand Visual and Audio Content:** Process images to identify products and suggest similar items, and transcribe audio messages to understand queries and respond with a generated audio message.
* **Handle General Business Inquiries:** Answer common questions about store hours, locations, shipping policies, and payment methods.
* **Connect to Tools:** Act as a central orchestrator, connecting to a Retrieval-Augmented Generation (RAG) system, a product database, and specialized multimodal databases for image and audio data.
* **Manage Memory:** Maintain both short-term conversational context and long-term memory to remember specific customer information and preferences.
* **Adapt to Customer Preferences:** Personalize interactions based on past behavior and stored preferences.

This project is not just a coding exercise; it's a comprehensive learning experience focused on building real-world, deployable products with a modular and scalable architecture.

## Project Scopes

The project is structured into clear, iterative scopes to guide development and learning.

* **Scope 1:** Foundational Q&A Agent: Build a robust, text-based agent that uses RAG to respond to general business inquiries and product-related questions.
* **Scope 2:** Multi-Modal Agent: Extend the agent to handle different types of input, such as images (for product recognition) and audio (for voice commands).
* **Scope 3:** Tool-Use and Actions: Enable the agent to perform actions by integrating with external tools and APIs (e.g., checking order status, browsing product catalogs).
* **Scope 4:** Advanced Memory and Context: Implement a more sophisticated memory system to allow the agent to remember past conversations and provide a more personalized experience.

## What You Will Learn

This project is more than just an agent; it's a comprehensive learning experience. By following along, you will gain hands-on experience with:

* Building scalable and robust APIs with Python and FastAPI.
* Containerizing applications and managing services with Docker and Docker Compose.
* Working with PostgreSQL as a vector and relational database.
* Implementing Retrieval-Augmented Generation (RAG) for grounded, up-to-date responses.
* Leveraging frameworks like LangGraph to simplify LLM application development.
* Integrating third-party services like Telegram and various LLM providers.
* Applying a production-ready layered architecture (Domain, Application, Infrastructure) to your projects.
* Debugging and troubleshooting a distributed AI application.

## Getting Started

Clone this repository: git clone https://github.com/your-username/your-repo-name.git

Follow the instructions in `INSTALLATION_AND_USAGE.md`.

## YouTube Series

This repository is the official code for our YouTube series on building this AI agent. Check out the channel to watch the videos and build the project step-by-step with us!