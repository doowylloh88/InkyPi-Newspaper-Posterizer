# InkyPi Newspaper Posterizer

*InkyPi Newspaper Posterizer* is a plug-in for [InkyPi](https://github.com/fatihak/InkyPi) that transforms newspaper headlines into an easy to read, stylized poster

**What it does:**

- **Front Page** — Turns today’s main headline into a poster in the style you choose  
- **Newspapers!** — Pick from 600+ newspapers sourced from Freedom Forum’s front page archive
- **What's your style?** — Includes 30+ built-in styles or vibes; add your own or delete existing ones from the UI  
- **AI Magic** — Requires a paid [Open API](https://platform.openai.com/) key to generate the final poster image 

- **How It Works** 
    - You pick a newspaper 
    - The plug-in fetches today's front page
    - An AI vision model analyzes it & extracts the main headline + a short blurb
    - Using your selected style, the image model generates a clean poster layout 


## Screenshot

![screenshot](https://github.com/doowylloh88/InkyPi-Newspaper-Posterizer/blob/main/docs/images/Skater_Headline.jpg)

## Installation

### Install

Install the plugin using the InkyPi CLI, providing the plugin ID & GitHub repository URL:

```bash
inkypi install doowylloh88 https://github.com/doowylloh88/InkyPi-Newspaper-Posterizer

```
**Requirements**

- **Open AI API** — You'll need a paid [Open API](https://platform.openai.com/) key to analyze the newspaper's front page and to generate an image

- **Flexible analysis** — You can also analyze the front page using [Groq / Llama Vision](https://platform.openai.com/) 
- OpenAI can be stricter with copyrighted content; Groq/Llama is often more forgiving ( and I just like the results better )
- Put the API keys in the .env file in the Inky Pi root directory

## Development-status

- Speaking of vibes, this plug-in was 100% created using vibe- coding & a lot of yelling at ChatGPT.  An actual coder should take over the project to maintain it

- I've updated the newspaper list to include over 600 newspapers sourced from [freedom forum]( https://frontpages.freedomforum.org)

- HUGE CAVEAT: I'm in developer mode ONLY. I don't have an e-ink screen yet.  They are sold out everywhere.  As I wait for stocks to resupply,  I'm creating plug-ins I want. I'm running off a Raspberry Pi 5.  Most of the image processing happens on the Open AI side, so it shouldn't slam a Pi Zero 2W any more than the other existing image plug-ins


## License

This project is licensed under the GNU public License


