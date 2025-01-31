import discord
import os
import json
import pandas as pd
import re

# Define paths to the files
excel_path = r"V:\NPI\Share\1.LAB OP\9. CNB Test Report\raw\LTP\charger storage test.xlsx"
json_path = r"V:\NPI\Share\1.LAB OP\9. CNB Test Report\raw\LTP\test_config_OP40804VNM.json"

# Load the JSON configuration file
with open(json_path, 'r') as file:
    config = json.load(file)

# Load the Excel file
df = pd.read_excel(excel_path, engine='openpyxl')

# Define the bot client with intents
intents = discord.Intents.default()
intents.guilds = True
intents.members = True
client = discord.Client(intents=intents)

# Load token from environment variable for security
TOKEN = os.getenv('DISCORD_BOT_TOKEN')


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

    # Find user by name
    user = discord.utils.get(client.get_all_members(), name="tom_le_96")
    if user:
        # Iterate through each row in the DataFrame and send the processed message
        for index, row in df.iterrows():
            step = row['step']
            action = row['action']

            # Find words within square brackets and replace them with values from JSON
            matches = re.findall(r'\[(.*?)\]', action)
            for match in matches:
                # Check in "battery_pack_discharge" first, then in "general"
                value = config.get("charging characteristics", {}).get(match) or config.get("general", {}).get(match)
                if value is not None:
                    action = action.replace(f'[{match}]', str(value))

            # Send the processed step and action to the user
            await user.send(f"Step: {step}, Action: {action}")
    else:
        print("User tom_le_96 not found.")


# Run the bot
client.run('MTI5MjY5NzYwMDM0ODI2MjQzMA.GbmEDn.IGSaYUU1u0SRUzbK45s1hL8b2FwwMs8czba1z8')
