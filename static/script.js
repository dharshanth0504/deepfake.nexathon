document.addEventListener("DOMContentLoaded", function() {
    let button = document.querySelector("button");
    button.addEventListener("mouseover", function() {
        this.style.backgroundColor = "#ff1744";
    });
    button.addEventListener("mouseleave", function() {
        this.style.backgroundColor = "#ff5722";
    });
});
