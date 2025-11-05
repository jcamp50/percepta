"""
Seed test channel with multiple long transcripts for testing.

This script creates realistic test data by inserting multiple transcript chunks
into the vector store for a test channel. These transcripts will be used for
RAG queries in multi-user testing.
"""

import asyncio
import sys
import codecs
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from py.memory.vector_store import VectorStore
from py.utils.embeddings import embed_text

# Test channel broadcaster ID (use broadcaster ID, not channel name)
TEST_CHANNEL_BROADCASTER_ID = "577427971"  # awinewofi5783

# Sample transcripts with diverse topics for testing
TRANSCRIPTS = [
    {
        "text": """Welcome to the stream everyone! Today we're playing a new indie game that just came out. 
        It's called "Mystic Realm" and it's a puzzle adventure game. The graphics are really beautiful, 
        and the soundtrack is amazing. Let me show you the gameplay mechanics. You can see here that 
        the character can interact with various objects in the environment. The puzzles are quite challenging 
        but fair. I've been playing for about an hour now and I'm really enjoying it. The story is intriguing 
        too - you're playing as a character who wakes up in a mysterious world with no memory of how they got there. 
        You need to solve puzzles to uncover the truth about what happened. The developers really put a lot of thought 
        into this game. The controls are smooth and responsive. I love how each puzzle teaches you something new 
        about the game mechanics. This is definitely going on my list of favorite indie games.""",
        "duration_seconds": 180,
    },
    {
        "text": """Okay chat, I know you've been asking about my setup. Let me break it down for you. 
        I'm using a RTX 4090 graphics card which handles everything beautifully. The CPU is an AMD Ryzen 9 7950X. 
        For streaming, I'm using OBS with NVENC encoding. My microphone is a Shure SM7B connected through a Focusrite 
        Scarlett audio interface. The camera is a Sony A7III. I've been using this setup for about six months now 
        and it's been rock solid. The streaming quality is great and I rarely have any issues. The only thing I 
        might upgrade soon is adding a second monitor for better workflow. But honestly, this setup does everything 
        I need. The audio quality is crisp and clear, the video quality is excellent, and the game performance is 
        top notch. I get consistent 144 FPS in most games at 1440p resolution. If anyone has questions about 
        specific parts of my setup, feel free to ask in chat.""",
        "duration_seconds": 240,
    },
    {
        "text": """So yesterday I was talking to some other streamers about the best ways to grow your channel. 
        The consensus was that consistency is key. You need to stream on a regular schedule so your audience knows 
        when to expect you. Also, engaging with chat is super important. Responding to messages, asking questions, 
        making viewers feel like they're part of the community. Another thing that helps is playing a variety of games 
        but also having a main game that you're known for. That way you attract different audiences but also build 
        a core community around your main content. Collaborating with other streamers is also valuable - doing raids, 
        hosting, or co-streaming can introduce you to new audiences. The streaming community is really supportive 
        overall. I've made a lot of friends through streaming and it's been an amazing journey. Starting out was tough 
        but sticking with it and being authentic has really paid off. The most important thing is to have fun and 
        be yourself.""",
        "duration_seconds": 200,
    },
    {
        "text": """This boss fight is absolutely brutal! I've been trying to beat this for like 30 minutes now. 
        The attack patterns are so unpredictable. Watch this - when he raises his left arm, he's about to do a 
        sweeping attack. You need to dodge to the right. Then when he glows red, that means he's about to charge. 
        The key is to stay mobile and never get greedy with attacks. One or two hits max, then back off. 
        The phase two is even worse - he starts spawning adds that you need to deal with while avoiding his attacks. 
        I think the strategy is to clear the adds first, then focus on the boss. But timing is everything. 
        If you're too slow clearing adds, more spawn. If you're too fast, you might not have enough time to 
        damage the boss before the next phase. This is definitely one of the hardest bosses in the game. 
        I've seen speedrunners do this fight perfectly, but I'm nowhere near that level. Just need to keep practicing 
        and learning the patterns. Every attempt teaches you something new about the fight mechanics.""",
        "duration_seconds": 220,
    },
    {
        "text": """Thank you so much for the subscription! That means a lot. And welcome to all the new followers 
        today. I'm really glad you're enjoying the stream. The chat has been super active today and I love it. 
        You guys are hilarious. That meme you just posted in chat had me cracking up. I should probably get back 
        to the game though before chat gets too rowdy. But seriously, thank you all for being here. Streaming 
        wouldn't be the same without this amazing community. The support you all show really motivates me to keep 
        creating content. Whether it's subscribing, following, or just hanging out in chat, it all means so much. 
        I'm trying to build a positive and welcoming community where everyone feels included. So if you're new here, 
        don't be shy - say hi in chat! We're all friendly here. Alright, let's get back to the game and see if 
        we can make some progress. I've been stuck on this level for a while but I think I'm close to figuring 
        it out. With chat's help, we'll get through it together.""",
        "duration_seconds": 150,
    },
    {
        "text": """The meta in this game has really shifted since the last update. The new character they added 
        is completely overpowered. Everyone is playing them now because they have such good mobility and damage. 
        The developers said they're going to nerf them in the next patch, but until then, you basically have to 
        play them or counter them. The counter strategy is to use characters with high burst damage to take them 
        out quickly before they can use their mobility. But it's tricky because they can dodge so easily. 
        I've been experimenting with different builds and I think I found one that works pretty well. 
        It focuses on crowd control and area denial. That way you can limit their movement options and force 
        them into bad positions. The balance team really needs to look at this though. It's not fun when one 
        character dominates the meta so hard. Makes the game feel stale. Hopefully the patch comes soon because 
        the game is much more enjoyable when there's variety in the characters people play.""",
        "duration_seconds": 190,
    },
    {
        "text": """I've been thinking about starting a new game series on the channel. Something different from 
        what I usually play. Maybe a story-driven game or a puzzle game. What do you think chat? I want to mix 
        things up a bit. The current game series has been going for a while now and I think it's time for 
        something fresh. I've been getting suggestions from chat about games to play and I've been looking into 
        them. There's this indie game that looks really interesting - it's called "Echoes of Eternity" and it's 
        got amazing reviews. It's a narrative adventure game with branching storylines. Your choices actually 
        matter and affect the outcome. That sounds really cool. I might start that next week. But I want to finish 
        the current game first. We're probably like 80% through it. Just a few more sessions and we'll be done. 
        Then we can move on to something new. But yeah, let me know in chat what you think. I'm always open to 
        suggestions from the community.""",
        "duration_seconds": 170,
    },
    {
        "text": """The tournament this weekend was insane! I can't believe how close the matches were. 
        Every single game came down to the wire. The final match was especially intense - it went to overtime 
        three times. I was on the edge of my seat the whole time. The players were absolutely incredible. 
        Some of the plays they made were just mind-blowing. That one clutch where the player had to 1v3 and 
        actually pulled it off? Unbelievable. The tournament format was really good too. Double elimination 
        bracket meant that even if you lost early, you could still make a comeback. That made for some amazing 
        storylines. The underdog team that came back from the losers bracket to win the whole thing was amazing. 
        I love watching competitive play because you can learn so much from watching the best players. 
        Their positioning, decision-making, and mechanical skill are all things to study. I'm definitely going 
        to try to incorporate some of what I saw into my own gameplay.""",
        "duration_seconds": 210,
    },
]


async def seed_test_channel():
    """Seed test channel with multiple transcripts."""
    print("=" * 80)
    print("Seeding Test Channel with Transcripts")
    print("=" * 80)
    print(f"Broadcaster ID: {TEST_CHANNEL_BROADCASTER_ID}")
    print(f"Channel: awinewofi5783")
    print(f"Number of transcripts: {len(TRANSCRIPTS)}")
    print("=" * 80)

    vector_store = VectorStore()
    start_time = datetime.now(timezone.utc) - timedelta(hours=2)  # Start 2 hours ago

    inserted_count = 0
    total_chars = 0

    for i, transcript_data in enumerate(TRANSCRIPTS):
        text = transcript_data["text"]
        duration = transcript_data["duration_seconds"]

        # Calculate timestamps for this chunk
        chunk_start = start_time + timedelta(seconds=i * duration)
        chunk_end = chunk_start + timedelta(seconds=duration)

        print(f"\n[{i+1}/{len(TRANSCRIPTS)}] Processing transcript chunk...")
        print(f"  Duration: {duration}s")
        print(f"  Text length: {len(text)} chars")
        print(f"  Time range: {chunk_start.strftime('%H:%M:%S')} - {chunk_end.strftime('%H:%M:%S')}")

        try:
            # Generate embedding
            print("  Generating embedding...")
            embedding = await embed_text(text)

            # Insert into vector store using broadcaster ID
            print("  Inserting into vector store...")
            transcript_id = await vector_store.insert_transcript(
                channel_id=TEST_CHANNEL_BROADCASTER_ID,
                text_value=text,
                start_time=chunk_start,
                end_time=chunk_end,
                embedding=embedding,
            )

            inserted_count += 1
            total_chars += len(text)
            print(f"  [OK] Inserted transcript ID: {transcript_id}")

        except Exception as e:
            print(f"  [X] Error inserting transcript: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Seeding Complete")
    print("=" * 80)
    print(f"Successfully inserted: {inserted_count}/{len(TRANSCRIPTS)} transcripts")
    print(f"Total characters: {total_chars:,}")
    print(f"Broadcaster ID: {TEST_CHANNEL_BROADCASTER_ID}")
    print(f"Channel: awinewofi5783")
    print("\nThis channel is now ready for testing!")
    print("=" * 80)

    return TEST_CHANNEL_BROADCASTER_ID


if __name__ == "__main__":
    asyncio.run(seed_test_channel())

