window.addEventListener("load", () => {
    console.log("hello");
    const canvas = document.querySelector("#canvas");
    const c = canvas.getContext("2d")

    canvas.height = window.innerHeight;
    canvas.width= window.innerWidth;

    let painting = false;

    function startPosition(e){
        painting = true;
        draw(e)
    }
    
    function finishedPosition(){
        painting = false;
        c.beginPath()
    }
    
    function draw(e){
        if (!painting) return;

        c.lineWidth = 10;
        c.linecap = "round";
        c.strokeStyle = "black";

        // c.beginPath();
        c.lineTo(e.clientX, e.clientY);
        c.stroke();
        // c.moveTo(e.clientX, e.cleintY);

        console.log(e.clientX, e.clientY)
    }

    // c.fillRect(100, 100, 100, 100)

    canvas.addEventListener("mousedown", startPosition);
    canvas.addEventListener("mouseup", finishedPosition);
    canvas.addEventListener("mousemove", draw);

    // c.beginPath();
    // c.moveTo(10, 20);
    // c.lineTo(100, 20);
    // c.strokeStyle = "#FFAACC";
    // c.lineWidth = 1;
    // c.stroke();
});

