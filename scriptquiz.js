const questions=[
    {
        question: "Which part of the Constitution of India deals with Fundamental Rights?",
        answers: [
            {text: "Part I", correct: false},
            {text: "Part III", correct: true},
            {text: "Part IV", correct: false},
            {text: "Part V", correct: false},
        ]
    },
    {
        question: "Who is considered the principal architect of the Constitution of India?",
        answers: [
            {text: "Dr. B.R. Ambedkar", correct: true},
            {text: "Mahatma Gandhi", correct: false},
            {text: "Jawaharlal Nehru", correct: false},
            {text: "Sardar Vallabhbhai Patel", correct: false},
        ]
    },
    {
        question: "Which article of the Constitution of India abolishes untouchability?",
        answers: [
            {text: "Article 14", correct: false},
            {text: "Article 19", correct: false},
            {text: "Article 21", correct: false},
            {text: "Article 17", correct: true},
        ]
    },
    {
        question: "The Directive Principles of State Policy are enshrined in which part of the Constitution of India?",
        answers: [
            {text: "Part I", correct: false},
            {text: "Part II", correct: false},
            {text: "Part IV", correct: true},
            {text: "Part V", correct: false},
        ]
    },
    {
        question: "Which of the following is not a Fundamental Right under the Constitution of India?",
        answers: [
            {text: "Right to Equality", correct: false},
            {text: "Right to Freedom of Religion", correct: false},
            {text: "Right to Property", correct: true},
            {text: "Right to Constitutional Remedies", correct: false},
        ]
    },
    {
        question: "The Preamble to the Constitution declares India as a:",
        answers: [
            {text: "Sovereign Socialist Secular Democratic Republic", correct: true},
            {text: "Sovereign Secular Democratic Republic", correct: false},
            {text: "Sovereign Democratic Republic", correct: false},
            {text: "Socialist Secular Democratic Republicg", correct: false},
        ]
    },
    {
        question: "Which amendment to the Constitution of India is known as the 'Mini Constitution'?",
        answers: [
            {text: "24th Amendment", correct: false},
            {text: "42nd Amendment", correct: true},
            {text: "44th Amendment", correct: false},
            {text: "86th Amendment", correct: false},
        ]
    },
    {
        question: "The Constitution of India was adopted by the Constituent Assembly on:",
        answers: [
            {text: "26th January 1950", correct: false},
            {text: "15th August 1947", correct: false},
            {text: "26th November 1949", correct: true},
            {text: "30th January 1950", correct: false},
        ]
    },
    {
        question: "Which schedule of the Constitution of India deals with the allocation of seats in the Rajya Sabha?",
        answers: [
            {text: "First Schedule", correct: false},
            {text: "Third Schedule", correct: false},
            {text: "Fourth Schedule", correct: true},
            {text: "Seventh Schedule", correct: false},
        ]
    },
    {
        question: "The concept of 'Judicial Review' in the Constitution of India is borrowed from which country?",
        answers: [
            {text: "United Kingdom", correct: false},
            {text: "United States of America", correct: true},
            {text: "Canada", correct: false},
            {text: "Ireland", correct: false},
        ]
    },
];
const questionElement=document.getElementById("question");
const answerButtons=document.getElementById("answer-buttons");
const nextButton=document.getElementById("next-btn");

let currentQuestionIndex=0;
let score=0;

function startQuiz(){
    currentQuestionIndex=0;
    score=0;
    nextButton.innerHTML="Next";
    showQuestion();
}
function showQuestion(){
    resetState();
    let currentQuestion=questions[currentQuestionIndex];
    let questionNo=currentQuestionIndex+1;
    questionElement.innerHTML=questionNo+". "+currentQuestion.question;
    currentQuestion.answers.forEach(answer => {
        const button=document.createElement("button");
        button.innerHTML=answer.text;
        button.classList.add("btn");
        answerButtons.appendChild(button);
        if(answer.correct){
            button.dataset.correct=answer.correct;
        }
        button.addEventListener("click", selectAnswer);
    });
}
function resetState(){
    nextButton.style.display="none";
    while(answerButtons.firstChild){
        answerButtons.removeChild(answerButtons.firstChild);
    }
}
function selectAnswer(e){
    const selectedBtn=e.target;
    const isCorrect=selectedBtn.dataset.correct === "true";
    if(isCorrect){
        selectedBtn.classList.add("correct");
        score++;
    }else{
        selectedBtn.classList.add("incorrect");
    }
    Array.from(answerButtons.children).forEach(button => {
        if(button.dataset.correct === "true"){
            button.classList.add("correct");
        }
        button.disabled = true;
    });
    nextButton.style.display="block";
}
function showScore(){
    resetState();
    questionElement.innerHTML=`You scored ${score} out of ${questions.length}!`;
    nextButton.innerHTML="Play Again";
    nextButton.style.display="block";
}
function handleNextButton(){
    currentQuestionIndex++;
    if(currentQuestionIndex<questions.length){
        showQuestion();
    }else{
        showScore();
    }
}
nextButton.addEventListener("click", ()=>{
    if(currentQuestionIndex<questions.length){
        handleNextButton();
    }else{
        startQuiz();
    }
})
startQuiz();