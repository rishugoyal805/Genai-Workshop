import os
import dotenv
from crewai import Agent, Task, Crew, LLM

dotenv.load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = LLM(model="gemini/gemini-1.5-flash", api_key=GEMINI_API_KEY, verbose=True)

weather_chatbot = Agent(
    name="Weather Chatbot",
    role="Basic Weather Responder",
    goal="Respond to weather questions using text prediction only.",
    backstory="Answer weather questions by generating text, without using live data.",
    llm=llm,
    verbose=True
)

if __name__ == "__main__":
    while True:
        city = input("\nEnter the city name to get weather (or type 'exit' to quit): ")
        if city.lower() == "exit":
            print("Exiting Weather Chatbot. Goodbye!")
            break

        task = Task(
            description=f"Tell me the weather in {city}",
            agent=weather_chatbot,
            expected_output=f"A text-based answer about {city} weather (not real-time)."
        )

        crew = Crew(agents=[weather_chatbot], tasks=[task])

        result = crew.kickoff()
        print("AI Agent Response:", result.raw)
