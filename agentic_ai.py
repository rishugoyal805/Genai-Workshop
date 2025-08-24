import os
import dotenv
from crewai import Agent, Task, Crew, LLM

dotenv.load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    verbose=True
)

weather_fetcher = Agent(
    name="Weather Fetcher",
    role="Data Retriever",
    goal="Fetch live or static weather information.",
    backstory="Specialized in gathering weather details from trusted sources or datasets.",
    llm=llm
)

weather_explainer = Agent(
    name="Weather Explainer",
    role="Simplifier",
    goal="Explain the fetched weather details in simple words.",
    backstory="Transforms raw weather data into human-friendly explanations.",
    llm=llm
)

if __name__ == "__main__":
    while True:
        city = input("\nEnter the city name to get weather (or type 'exit' to quit): ")
        if city.lower() == "exit":
            print("Exiting Weather Chatbot. Goodbye!")
            break

        # Define Tasks dynamically based on input
        task1 = Task(
            description=f"Get the weather details for {city}.",
            agent=weather_fetcher,
            expected_output=f"Raw weather d ata for {city}."
        )

        task2 = Task(
            description="Explain the weather details in simple words.",
            agent=weather_explainer,
            expected_output="A user-friendly weather explanation."
        )

        # Create Crew
        crew = Crew(
            agents=[weather_fetcher, weather_explainer],
            tasks=[task1, task2]
        )

        # Run the Crew
        result = crew.kickoff()
        print("\nAgentic AI Response:", result)
