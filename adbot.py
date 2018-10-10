import os
import sys
import json
import requests
from flask import Flask, request,render_template, redirect
from bot.run import run_bot,run_bot_pre,action_train_pre
from templates.forms import InputForm
from bot.secret_weapon.bot_seq2seq import bot_predict_reply,pre_train_bot,bot_predict_sim_reply
#from bot.secret_weapon.question_siamese import siamese_train,return_highsim_sentence
from bot.secret_weapon.sim_sentence_api import simhash_x,simhash_reply
from bot.secret_weapon.bot_doc2vec_models import doc2vec_sim_reply,doc2vec_train


app = Flask(__name__)
app.config['SECRET_KEY'] = '180709aigj'

@app.route('/', methods=['GET'])
def verify():
    """verify"""
    # when the endpoint is registered as a webhook, it must echo back
    # the 'hub.challenge' value it receives in the query arguments
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == os.environ["VERIFY_TOKEN"]:
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200

    return "go to http://127.0.0.1:5000/test", 200

@app.route('/test',methods=['GET', 'POST'])
def test():
    """test UI"""
    form = InputForm()
    if form.validate_on_submit():
        reply = run_bot(form.input_data.data)
        input_text = form.input_data.data
        form.input_data.data = ""
        return render_template('index.html', reply = reply, form = form, input_text = input_text)
    return render_template('index.html', form = form)

def log(message):
    """function for logging"""
    if message:
       print(str(message))
       sys.stdout.flush()
    else:
       print("NULL")
       sys.stdout.flush()

if __name__ == '__main__':
    #app.debug = True
    #doc2vec_train()
    #run_bot_pre()
    #pre_train_bot()
    
    #siamese_train()
    #action_train_pre()
    #sentence = '高血压心脏病喝什么茶最好'
    sentence = '血脂稠头晕高血压怎么食疗？'
    print(bot_predict_reply(sentence))
    #sentence2 = '高血压？怎么办失眠'
    
    #print(bot_predict_sim_reply(str(sentence2)))
    #print(simhash_reply(str(sentence2)))
    '''
    sim_sentence = simhash_x(sentence2)
    print(sim_sentence)
    print(bot_predict_reply(str(sim_sentence)))
    print('**************************')
    '''
    #sim_sent = doc2vec_sim_reply(sentence2)
    #print(sim_sent)
    #print(bot_predict_reply(str(sim_sentence)))
    app.run()#
