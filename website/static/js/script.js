function formatDateTime(Chosenday) {
    const weekday = new Array(7);
    weekday[0] = "Sunday";
    weekday[1] = "Monday";
    weekday[2] = "Tuesday";
    weekday[3] = "Wednesday";
    weekday[4] = "Thursday";
    weekday[5] = "Friday";
    weekday[6] = "Saturday";
  
    let day = weekday[Chosenday.getDay()];
  
    return `${day}`
    // ${Chosenday.getHours()}:${Chosenday.getMinutes()}`;
  }

  let now = new Date()  
  let curTime = formatDateTime(now);
  let day = document.querySelector(".today__date p");
  day.innerHTML = curTime;
  let time = document.querySelector(".time");
  time.innerHTML = `${now.getHours()} : ${now.getMinutes()}`;
  
  function formatDate(Chosenday) {
    const months = new Array(12);
    months[0] = "January";
    months[1] = "February";
    months[2] = "March";
    months[3] = "April";
    months[4] = "May";
    months[5] = "June";
    months[6] = "July";
    months[7] = "August";
    months[8] = "September";
    months[9] = "October";
    months[10] = "November";
    months[11] = "December";
  
    let month = months[Chosenday.getMonth()];
  
    return `, ${month} ${Chosenday.getDate()}, ${Chosenday.getFullYear()}`;
  }
  
  let curDate = document.querySelector(".date");
  curDate.innerHTML = formatDate(now);

var x = document.querySelector(".temperature__today small");
function getLocation() {
    navigator.geolocation.getCurrentPosition(showPosition);
}

function showPosition(position) {
    x.innerHTML = `${Math.round(position.coords.latitude, 5)}N - ${Math.round(position.coords.longitude, 5)}E`;
}
getLocation()

$("#switch").on('click', function () {
  if ($("body").hasClass("fire-off")) {
    $("body").removeClass("fire-off");
    $("#switch").removeClass("switched");
  }
  else {
    $("body").addClass("fire-off");
    $("#switch").addClass("switched");

  }
});