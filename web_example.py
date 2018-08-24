import holdem
import agent

SERVER_URI = r"ws://poker-training.vtr.trendnet.org:3001" #TW test
#SERVER_URI = r"ws://poker-training.vtr.trendnet.org:3001" #TW warm-up AI contest
# SERVER_URI = r"ws://allhands2018-beta.dev.spn.a1q7.net:3001" # beta
# SERVER_URI = r"ws://allhands2018-training.dev.spn.a1q7.net:3001" # training

#https://en.wikipedia.org/wiki/The_Hitchhiker%27s_Guide_to_the_Galaxy_(novel)
# name="Enter Your Name Here"
name="helloworld"
#model = agent.allRaiseModel()
model = agent.dqnModel_run()

# while True: # Reconnect after Gameover
client_player = holdem.ClientPlayer(SERVER_URI, name, model, debug=True, playing_live=True)
client_player.doListen()