document.addEventListener('click', classifyText, true);

var bayes = new classifier.Bayesian();

//console.log(glossary);

function classifyText(eventData) {
	var target = eventData.target;
	console.log(target.innerText);
}