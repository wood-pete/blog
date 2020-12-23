window.onscroll = function() {scrollFunction()};

function scrollFunction() {
  if (document.body.scrollTop > 100 || document.documentElement.scrollTop > 100) {
    document.getElementById("myHeader").style.height = "100px";
  } else {
    document.getElementById("myHeader").style.fontSize = "150px";
  }
}
