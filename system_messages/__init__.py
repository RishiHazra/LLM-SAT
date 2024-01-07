# system messages for different ablations
from system_messages.menu import system_msg_menu
from system_messages.ablation_sat import system_msg_sat
from system_messages.ablation_translate import system_msg_translate

names = {'menu': system_msg_menu,
         'sat': system_msg_sat,
         'translate': system_msg_translate}