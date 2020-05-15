import aiml
import os

sessionID = 12345

kernel = aiml.Kernel()

sessionData = kernel.getSessionData(sessionID)

name = kernel.getPredicate("name", sessionID)

if os.path.isfile("bot_brain.brn"):
    x = raw_input("Load from the existing .brn file? (Enter y/n): ")
    if(x=='Y' or x=='y'):
        kernel.bootstrap(brainFile = "bot_brain.brn")
    else:
        kernel.bootstrap(learnFiles = "std-startup.xml", commands = "load aiml b")

else:
    kernel.bootstrap(learnFiles = "std-startup.xml", commands = "load aiml b")
print("========================================================================\nTHIS BOT WILL HELP YOU CHOOSE MOVIES!\n===============================================================\n")
print("Type save at any point to save a .brn file\nType bye/quit/exit to leave\nTry to not wander too far from the point")

while True:
    message = raw_input(">>>")
    if message == "quit" or message == "exit" or message == "bye":
        break
    elif message=="save":
        kernel.saveBrain("bot_brain.brn")
    else:
        bot_response = kernel.respond(message, sessionID)
        print(bot_response)
