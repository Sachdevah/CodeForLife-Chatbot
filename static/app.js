class Chatbox{
    //constructor which gets activted everytime we create a chatbox instance
    constructor(){
        this.args={
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            FontChangeButton: document.querySelector('.Font_Button'),
            ThemeChangeButton: document.querySelector('.Theme_Button'),
            EmailUsButton: document.querySelector('.Email_button'),
            FeedbackButton:  document.querySelector('.ProvideFeedback'),

        }
        this.state=false;
        this.messages=[];
    }

    //displaying all these content on html page
    display(){
        const {openButton,chatBox,sendButton,FontChangeButton,ThemeChangeButton, EmailUsButton, FeedbackButton}=this.args;
        //action listener for chatbot togglestate and send button
        openButton.addEventListener('click',()=>this.toggleState(chatBox))
        sendButton.addEventListener('click',()=>this.onsendButton(chatBox))

        //action listener for menu bar buttons
        FontChangeButton.addEventListener('click',this.changeFont)
        ThemeChangeButton.addEventListener('click',this.changeTheme)
        EmailUsButton.addEventListener('click',this.emailUs)
        FeedbackButton.addEventListener('click',this.GiveFeedback)

        const node=chatBox.querySelector('input');
        node.addEventListener("keyup",({key})=>{
            if(key==="Enter"){
                this.onsendButton(chatBox)
            }
        })


    }

    //mail to feature to redirect user to thier mail box with pre-filled "email to" feild
    emailUs() {
        var email = document.createElement("a");
        email.href = "mailto:laura.cumming@ocado.com";
        email.click();
    }

    //feedback button to get user feedback about the chatbot system over email
    GiveFeedback() {
        var email = document.createElement("a");
        email.href = "mailto:dioni.zhong@ocado.com";
        email.click();
    }


    //changing font size feature
    changeFont(){
        var fon=document.body;
        if (fon.style.fontSize == "small") {
            fon.style.fontSize = "medium";
        }
        else if(fon.style.fontSize == "medium"){
            fon.style.fontSize = "large";
        }
        else{
            fon.style.fontSize = "small";
        }
    }

    //changing theme feature
    changeTheme(){
        var col1=document.getElementById("chatbox_id");
        var col2=document.getElementById("chat_id");
        if (col1.style.background == "white") {

            col1.style.background="black";
            col2.style.background="darkgray";

        }
        else {

            col1.style.background="white";
            col2.style.background="lightgrey";

        }
    }


    //the chatbot pop-up window
    toggleState(Chatbox){
        this.state=!this.state;

        //show or hides the box
        if(this.state){
            Chatbox.classList.add('chatbox--active')
        }
        else{
            Chatbox.classList.remove('chatbox--active')
        }
    }

    //send button to send the message from user side
    onsendButton(chatbox){
        var textField=chatbox.querySelector('input');
        let text1=textField.value
        if(text1===""){
            return;
        }
        let msg1={name:"User", message: text1}
        this.messages.push(msg1);

        fetch($SCRIPT_ROOT+'/predict',{

            method: 'POST',
            body: JSON.stringify({message:text1}),
            mode: 'cors', 
            headers:{
                'Content-Type': 'application/json'
            },
        })
        .then(r=>r.json())
        .then(r=>{
            let msg2={name: "Chatty", message: r.answer};
            this.messages.push(msg2);
            this.updateChatText(chatbox)
            textField.value=''
        }).catch((error)=>{
            console.error('Error:',error);
            this.updateChatText(chatbox)
            textField.value=''
        });

    }

    //to get chatbot response to display
    updateChatText(chatbox){
        var html='';
        
        this.messages.slice().reverse().forEach(function(item){

            if (item.name==="Chatty") {
                html+='<div class="messages__item messages__item--visitor">'+item.message+'</div>'

            }
            else{
                html+='<div class="messages__item messages__item--operator">'+item.message+'</div>'

            }
        });
        const chatmessage=chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML=html;
    }

}

const ct=new Chatbox();
ct.display();

