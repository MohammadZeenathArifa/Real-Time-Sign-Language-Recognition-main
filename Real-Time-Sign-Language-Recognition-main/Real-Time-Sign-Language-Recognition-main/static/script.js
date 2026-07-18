function startCamera(){

    const cam = document.getElementById("cameraFeed");

    cam.src = "/video";
    cam.style.display = "block";

    document.getElementById("status").innerText =
    "Interpreter is Active";

    detectSign();
}

function detectSign(){

    setInterval(async ()=>{

        const response = await fetch("/predict");

        const data = await response.json();

        document.getElementById("prediction").innerText =
        "Detected Sign: " + data.prediction;

    },700);

}