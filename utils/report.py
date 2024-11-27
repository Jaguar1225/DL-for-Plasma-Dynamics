import os
import zipfile
from tqdm import tqdm
import smtplib
from email.message import EmailMessage

def zip_file(zip_filename,image_folder):
    if os.path.isdir(zip_filename):
        pass
    else:
        with zipfile.ZipFile(zip_filename,'w') as zipf:
            for root,dirs,files in os.walk(image_folder):
                for file in files:
                    if file.endswith((".png",".jpg",".jpeg")):
                        zipf.write(os.path.join(root,file),file)

# 이메일 메시지 생성
def report_to_mail(subject,sender_email,password,receiver_email, file_list = None, contents= None):

    # 첨부 파일 추가
    if file_list:
        if len(file_list) > 5:
            for i in tqdm(range(0,len(file_list),5)):
                msg = EmailMessage()
                msg['Subject'] = subject + f'({i+5}/{len(file_list)})'
                msg['From'] = sender_email
                msg['To'] = receiver_email
                if contents == None:
                    msg.set_content("Here is the loss data.")
                else:
                    msg.set_content(contents)
                sub_list = file_list[i:i+5]
                for file_name in sub_list:
                    with open(file_name,'rb') as file:
                        file_data = file.read()
                        msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(sender_email, password)
                    smtp.send_message(msg)
            print("Email sent successfully!")

        else:
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = receiver_email
            if contents == None:
                msg.set_content("Here is the loss data.")
            else:
                msg.set_content(contents)
            try:
                for file_name in tqdm(file_list):                
                    with open(file_name,'rb') as file:
                        file_data = file.read()
                        msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:

                    smtp.login(sender_email, password)
                    smtp.send_message(msg)
                    print("Email sent successfully!")
            except:
                pass
    else:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email
        if contents == None:
            msg.set_content("Here is the loss data.")
        else:
            msg.set_content(contents)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, password)
            smtp.send_message(msg)
            print("Email sent successfully!")