---
layout: post
title:  "Discord Bot with XML parsing"
categories: Python
---
This is a simple discord bot script (created together with [Barbaros Becet][barb]) which sends the current day's cafeteria menu to a specific channel.
The first part of the script parses the xml, finds the entry for the current day, checks if the category is available (not closed), appends the dish title to the message string, which will be sent by the discord bot.
The second part connects the discord client (the bot) and sends the message.
To run the script automatically on weekdays, I used scheduled [Github Actions][actions] for python applications.

Our university cafeteria posts the week's menu as an xml file in the following simplified format:
{% highlight xml %}
<menu>
    <day timestamp="1613948400">
        <item>
            <category>Vegeterian</category>
            <title>Pizza</title>
            <price>3</price>
        </item>        
        <item>
            <category>BBQ</category>
            <title>Chicken</title>
            <price>10</price>
        </item>
    </day>
    <day timestamp="1614034800">
        <item>
            <category>Vegeterian</category>
            <title>Pasta</title>
            <price>3</price>
        </item>        
        <item>
            <category>BBQ</category>
            <title>Closed</title>
            <price>0</price>
        </item>
    </day>
</menu>
{% endhighlight %}

my_bot_script.py
{% highlight python %}
import urllib.request
import xml.etree.ElementTree as ET
import datetime 

# open the url and read it's content
xml_file = urllib.request.urlopen('http://www.example.com/menu.xml')

#read the xml and store it as a string in data
data = xml_file.read()
xml_file.close()

root = ET.fromstring(data)
assert(root.tag == 'menu')

# find the menu entry for today 
today = datetime.date.today()
today_root = None
for child in root:
    child_timestamp = child.attrib['timestamp']
    child_timestamp_ = datetime.datetime.fromtimestamp(int(child_timestamp)).date()
    if child_timestamp_ == today:
        today_root = child
        break
    
closed_strings = ["Closed", "Not available"]

# find all available dishes and append them to a message string
message = ""
for item in today_root:
    category = item.find('category').text
    title = item.find('title').text
    if title not in closed_strings:
        message = message + "***"+category+"***" + '\n' + title + '\n'
        print(message)
        
if message == "":
    exit()

# Discord part: connect to the bot client using the bot token (should not be public) 
# and send the message to the specified channel
import discord

TOKEN = 'your-bot-token'
CHANNEL = 'your-channel'
client = discord.Client()


@client.event
async def on_ready():
    for channel in client.get_all_channels():
        if channel.name == CHANNEL:
            await channel.send(message)
    await client.close()
            
client.run(TOKEN)

{% endhighlight %}

This is the github action yaml file. It is mostly auto-generated in github. This workflow automatically runs the script from monday to friday at 8 am.

{% highlight yaml %}
name: Python application

on:
  schedule:
    - cron: '0 8 * * 1-5'

jobs:
  build:

    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python 3.9
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -U discord.py
    - name: Run bot script
      run: |
        python my_bot_script.py
        
{% endhighlight %}
Check out this [tutorial][tutorial-py] for more info on how to create a bot for discord and add it to your server and [ElementTree API][xml-api] for more info on how to parse your xml template.

[tutorial-py]: https://realpython.com/how-to-make-a-discord-bot-python/
[xml-api]: https://docs.python.org/3/library/xml.etree.elementtree.html
[actions]: https://docs.github.com/en/actions
[barb]: https://github.com/barbarosbecet
