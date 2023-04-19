// common.js

function getUrlParams() {
    // Ref: https://stackoverflow.com/questions/4656843/get-querystring-from-url-using-jquery/4656873#4656873
    var vars = [], hash;
    var hashes = window.location.href.slice(window.location.href.indexOf('?') + 1).split('&');
    for(var i = 0; i < hashes.length; i++) {
        hash = hashes[i].split('=');
        vars.push(hash[0]);
        vars[hash[0]] = hash[1];
    }
    return vars;
}

function setUrlQueryValue(paramKey, paramVal) {
    try {
        var href = new URL(location.href);
        // href.searchParams.set(paramKey, encodeURI(paramVal));  // NOTE: double encoding!!
        href.searchParams.set(paramKey, paramVal);  // Encoded, like with encodeURI(), except + instead of %20 which is annoying
        window.history.pushState("", "", href);
        return encodeURIComponent(paramVal);  //.replace(/%20/g, '+');   // TODO: review, replace not needed?
    } catch(err) { }
}

function deleteUrlQueryKey(paramKey) {
    try {
        var href = new URL(location.href);
        href.searchParams.delete(paramKey);
        window.history.pushState("", "", href);  // TODO: review
    } catch(err) { }
}

// Options: s (style), l (logo), d (debug), b (business)
function getUrlQuery(k) {
    try {
        var params = getUrlParams();
        if (k in params) {
            // NOTE: plus_fixed replace to solve href.searchParams.set encoding as described above.
            //       https://stackoverflow.com/questions/4535288/why-doesnt-decodeuriab-a-b/4535319#4535319
            var plus_fixed = (params[k]+'').replace(/\+/g, '%20')
            return decodeURIComponent(plus_fixed); } else { return "";
        }
    } catch(err) {
        return "";
    }
}


function toTitleCase(str) {
  return str.replace(
    /\w\S*/g,
    function(txt) {
      return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
    }
  );
}

// https://stackoverflow.com/questions/400212/how-do-i-copy-to-the-clipboard-in-javascript/33928558#33928558
function copyToClipboard(text) {
    if (window.clipboardData && window.clipboardData.setData) {
        // Internet Explorer-specific code path to prevent textarea being shown while dialog is visible.
        return window.clipboardData.setData("Text", text);

    }
    else if (document.queryCommandSupported && document.queryCommandSupported("copy")) {
        var textarea = document.createElement("textarea");
        textarea.textContent = text;
        textarea.style.position = "fixed";  // Prevent scrolling to bottom of page in Microsoft Edge.
        document.body.appendChild(textarea);
        textarea.select();
        try {
            return document.execCommand("copy");  // Security exception may be thrown by some browsers.
        }
        catch (ex) {
            console.warn("Copy to clipboard failed.", ex);
            return prompt("Copy to clipboard: Ctrl+C, Enter", text);
        }
        finally {
            document.body.removeChild(textarea);
        }
    }
}

function scrollToBottom() {
    // TODO: investigate smooth: https://stackoverflow.com/questions/42261524/how-to-window-scrollto-with-a-smooth-effect
    window.scrollTo(0, $(document).height());
}


$(document).ready(function () {
    $("#menu-icon").click(function(){
        // https://www.w3schools.com/howto/howto_js_sidenav.asp
        $('#sidenav').width("200px");
    });

    $("#menu-close-btn").click(function(){
        $('#sidenav').width("0px");
    });

    $("#container").click(function(){
        $('#sidenav').width("0px");
    });
});


// ask.js

const app = "awepy-ask-202304";
const baseAppUrl = "https://www.awesomepython.org";
const authkey = "58906b3d-4a8a-4712-89c8-56446590ef73";
const predictionUrl = "https://chat-api.infocruncher.com/predict/";
const randomInputs = [
        // "How do I format a date string in Python with format YYYMMDD?",
        "Data Engineer",
        "Corporate Lawyer",
        "Fireman",
        "Technical Program Manager",
        "Clinical Exercise Physiologist ",
        "Site Reliability Engineer",
        "Sales Development Representative",
        "Platform Engineer",
        "Prompt Engineer",
        "Growth Specialist",
        "Claims Adjuster",
        "Machine Learning Engineer",
        "Sustainability Manager",
        "Enterprise Account Executive",
        "Cyber Security Analyst",
        "Crime Analyst",
        "Data Engineer",
        "Job Coach",
        "Cloud Engineer",
        "Customer Success Manager",
        "Client Associate",
        "Business Development Representative",
        "Health Assistant",
        "Service Desk Engineer",
        "Delivery Consultant",
        "Cyber Security Engineer",
        "Finance Associate",
        "Product Designer",
        "Technology Project Manager",
        "Housekeeper",
        "Food specialist",
        "Pharmacy specialist",
        "Tax consultant",
        "Python developer",
        "Software engineer",
        "JavaScript developer",
        "Salesperson",
        "Registered nurse",
        "Java Software Engineer",
        "Virtual Event Planner",
        "Climate Change Analyst",
        "Personal Data Broker",
        "Augmented Reality Designer",
        "Quantum Computing Engineer",
        "Cybersecurity Analyst",
        "Health Informatics Specialist",
        "Digital Currency Advisor",
        "Robotics Technician",
        "Urban Agriculture Specialist",
    ]

var rndIndex = Math.floor(Math.random() * randomInputs.length);
var predict_count = 0;

function populateQueryShareLink(businessIdea) {
    var shareLinkValue = baseAppUrl;
    if (businessIdea != "") {
        shareLinkValue += "?b=" + encodeURIComponent(businessIdea);
    }
//    if (businessIdea != "" && style != "") {
//        shareLinkValue += "&s=" + encodeURIComponent(style);  // TODO: need to select radio button of style
//    }
    $("#copy-idea-button").data("link", shareLinkValue);
}

var bar = new ProgressBar.Line(
    '#progress',
    {
        easing: 'easeInOut',
        duration: 13000,
        color: '#3776ab',
    });

function showProgressBar() {
    // TODO: dynamically set duration based on style
    $('#progress').show();
    bar.set(0.01);
    bar.animate(1, {}, function() {
        var existingText = $("#response").html();
        $("#response").html(existingText + "<p><i>Doing some fancy formatting, soooooooooooo close now!...</i></p>");
    });
}

function hideProgressBar() {
    bar.set(1);
    $('#progress').hide("slow");
    bar.set(0);
}

$(document).ready(function () {
    var businessIdea = getUrlQuery("b");
    var style = getUrlQuery("s");
    var load = getUrlQuery("load");
    console.log("User querystring businessIdea: " + businessIdea);
    console.log("User querystring style: " + style);
    console.log("User querystring load: " + load);

    $("#input-text").focus();
    populateQueryShareLink(businessIdea);

    if (load != "") {
        // Load a previously generated report to display
        $("#input-text").val("load:" + load);
        // setTimeout(function(){ $('#input-button').click()}, 100);
        processReport();

    } else if (businessIdea != "") {
        // Nothing to load, user supplied a business idea in querystring
        $("#input-text").val(businessIdea);
        populateQueryShareLink(businessIdea);

//    } else {
//        // Autopopulate input with a random input
//        var randomElement = randomInputs[Math.floor(Math.random() * randomInputs.length)];
//        $("#input-text").val(randomElement);
//        populateQueryShareLink(randomElement);
    }

    $("#clear-button").click(function(){
        deleteUrlQueryKey("b");
        $("#input-text").val("")
        $("#input-text").focus();
        populateQueryShareLink("");
    });

    $("#random-button").click(function(){
        rndIndex += 1;
        if (rndIndex >= randomInputs.length) {
            rndIndex = 0;
        }
        var randomElement = randomInputs[rndIndex];
        $("#input-text").val(randomElement);
        populateQueryShareLink(randomElement);
    });

    $("#view-option-info").click(function(){
        $("#share-info").hide();
        $("#option-info").toggle();
        scrollToBottom();
        return false;
    });

    $("#view-option-info-intext").click(function(){
        $("#share-info").hide();
        $("#option-info").toggle();
        scrollToBottom();
        return false;
    });

    $("#view-share-info").click(function(){
        $("#option-info").hide();
        $("#share-info").toggle();
        scrollToBottom();
        return false;
    });

    $("#input-text").keypress(function (e) {
      if (e.which == 13) {
        $("#input-button").click();
        return false;
      }
    });

    $("#copy-idea-button").click(function() {
        var link = $(this).data("link");
        console.log("Copied idea link: " + link);
        copyToClipboard(link);

        var button = $(this);
        button.addClass("button-clicked");
        setTimeout(function() {
          button.removeClass("button-clicked");
        }, 200);
    });

    $("#copy-report-button").click(function() {
        var link = $(this).data("link");
        console.log("Copied report link: " + link);
        copyToClipboard(link);

        var button = $(this);
        button.addClass("button-clicked");
        setTimeout(function() {
          button.removeClass("button-clicked");
        }, 200);
    });

    $("#input-button-bottom").click(function(){
        document.documentElement.scrollTop = 0;
        return false;
    });

    $(".input-button-class").click(function() {  // NOTE: match class as there are 2 buttons
        processReport();
    });

    var start = new Date().getTime();
    $.ajax({
       type: "POST",
       url: predictionUrl,
       data: '{"app": "' + app + '", "authkey": "' + authkey + '", "query": "PrimeLambda", "mode": "PrimeLambda"}',
       tryCount : 0,
       retryLimit : 1,
       success: function(data)
       {
            predict_count++;
            var end = new Date().getTime();
            var time = end - start;
            var debugInfo = "\nPRIME SUCCESS:\nClient timing (ms): " + time + "\n"
                             + "Retries: "+this.tryCount+"\n"
                             + "Success Reponse:\n"
                             + JSON.stringify(data, null, "  ");
            console.log(debugInfo);
       },
       error: function(data)
       {
           this.tryCount++;
           if (this.tryCount <= this.retryLimit) {
               $("#response").html("Retry " + this.tryCount + "...");
               $.ajax(this);
               return;
           }
           var end = new Date().getTime();
           var time = end - start;

           var debugInfo = "\nPRIME ERROR:\nClient timing (ms): " + time + "\n"
                            + "Retries: "+this.tryCount+"\n"
                            + "Error Reponse:\n"
                            + JSON.stringify(data, null, "  ");
           console.log(debugInfo);
       },
     });
});


function processReport() {
    var raw_input = $("#input-text").val();
    var input = DOMPurify.sanitize(raw_input.replace(/(\r\n|\n|\r)/gm, " "));
    if (input.length == 0) {
        $("#response").html("You haven't entered any career dreams in the textbox above!<br /><br />"
            + "Enter the unicorn career do you want build (or try experimenting by clicking the 'Random Career' button) and give it another whirl");
        return;
    }

    var max_query_length = 250; // TODO: sync with app.py
    if (input.length > max_query_length) {
        $("#response").html("Can you squish your wild dream into " + max_query_length + " characters or less, and try again?<br />");
        return;
    }

    var mode = "section_dict";
    var style = getUrlQuery("s");  // NOTE: see `style_dict` in code for list.
    if (style == "") {
        style = $('input:radio[name="response_style"]:checked').val();
    }
    var logo = getUrlQuery("l");
    if (logo == "") {
        logo = $('input:radio[name="include_logo"]:checked').val();
    }
    var debug = getUrlQuery("d");

    var predictionUrlPayloadObject = {
        "app": app,
        "authkey": authkey,
        "query": input,
        "mode": mode,
        "style": style,
        "logo": logo,
        "debug": debug
    }
    var predictionUrlPayload = JSON.stringify(predictionUrlPayloadObject);
//        console.log("predictionUrlPayload: " + predictionUrlPayload);

    // Update URL with input value for sharing
    var paramVal = setUrlQueryValue("b", input);

    // Populate share link
    populateQueryShareLink(input);

    // Handle UI for input processing
    showProgressBar();  // New progress bar
    var originalInputButtonHtml = $("#input-button").html();
    var spinningInputButtonHtml = '<i class="fa fa-spinner fa-spin"></i> ' + originalInputButtonHtml;
    $("#input-button").html(spinningInputButtonHtml);
    $("#input-button").prop('disabled', true);
    $("#input-button-bottom").html(spinningInputButtonHtml);
    $("#input-button-bottom").prop('disabled', true);

    var style_text = style.length > 0 && style != "general" ? " in the style of " + toTitleCase(style) : "";  // Sync with business_report.py
    var running_text = "";
    if (input.startsWith("load:")) {
        // Loading a saved report
        running_text += "<p><em>Loading your personalised unicorn business plan:</em><p>";
    } else {
        // Building a saved report
        running_text += "<p><em>Building your personalised career plan" + style_text + ":</em><p>";
        running_text += "<h2>" + input + "</h2>";
    }

    running_text += "<p><i>Working on it now...almost done...</i></p>";
    if (predict_count == 0) {
       $("#response").html(running_text);  // NOTE: possible Lambda fn isn't primed yet.
    } else {
       $("#response").html(running_text);  // NOTE: Lambda should be primed.
    }

    var start = new Date().getTime();
    $.ajax({
       type: "POST",
       url: predictionUrl,
       data: predictionUrlPayload,
       tryCount : 0,
       retryLimit : 1,
       success: function(data)
       {
           // Reset UI after processing (success)
           hideProgressBar();  // New progress bar
           $("#input-button").html(originalInputButtonHtml);
           $("#input-button").prop('disabled', false);
           $("#input-button-bottom").html(originalInputButtonHtml);
           $("#input-button-bottom").prop('disabled', false);

           predict_count++;
           var end = new Date().getTime();
           var time = end - start;
           var chat_response = data["result"]["chat_response"];
           var request_id = data["request_id"];
           var request_id_loaded = data["request_id_loaded"];
           var query = data["query"];

           // console.log("chat_response: " + chat_response);
           console.log("request_id: " + request_id);
           console.log("request_id_loaded: " + request_id_loaded);
           console.log("query: " + query);

           var permalinkHref = "/ask.html?load=" + request_id + "&desc=" + encodeURIComponent(query);
           var permalink = '<a href="' + permalinkHref + '">Permalink to this plan</a>';
           var html_response = "";
           style_text = (style == "general") ? "" : " in the style of " + toTitleCase(style);
           if (request_id_loaded != null && query != null) {
                // Request loaded an existing business plan, update things for this case
                $("#input-text").val(query);
                $("#share-info").show();
                deleteUrlQueryKey("b");
                // html_response += "<i>Loaded your personalised unicorn business plan" + style_text + ":</i><br /><br />";
                html_response += "<i>Loaded your unique, personalised career plan" + style_text + " (" + permalink + "):</i><br /><br />";

           } else {
                // html_response += "<i>Created your personalised unicorn business plan" + style_text + ":</i><br /><br />";
                html_response += "<i>Created your unique, personalised career plan" + style_text + " (" + permalink + "):</i><br /><br />";
           }

           var markdown_as_sanitized_html = DOMPurify.sanitize(marked.parse(chat_response));  // Convert markdown chat_response into sanitized html
           html_response += markdown_as_sanitized_html;
           html_response += permalink + '<br /><br />';
           html_response += '<a href="#" onclick="document.documentElement.scrollTop = 0; return false;">Back to top</a>';

           $("#response").html(html_response);
           var copyReportPermalink = baseAppUrl + permalinkHref;
           $("#copy-report-button").data("link", copyReportPermalink);
           $("#copy-report-button").show();
           $("#copy-idea-button").hide();
           $("#share-info").show();

//               var debugInfo = "\nDEBUG INFO:\nClient timing (ms): " + time + "\n"
//                                                    + "Retries: "+this.tryCount+"\n"
//                                                    + "Success Reponse:\n"
//                                                    + JSON.stringify(data, null, "  ");
//               console.log(debugInfo);
       },
       error: function(data)
       {
           this.tryCount++;
           if (this.tryCount <= this.retryLimit) {
               $("#response").html("Hmmm, server is a bit busy, let me try that again for you...retry " + this.tryCount + " of " + this.retryLimit + "...");
               $.ajax(this);
               return;
           }
           var end = new Date().getTime();
           var time = end - start;

           // Reset UI after processing (error)
           hideProgressBar();  // New progress bar
           $("#input-button").html(originalInputButtonHtml);
           $("#input-button").prop('disabled', false);
           $("#input-button-bottom").html(originalInputButtonHtml);
           $("#input-button-bottom").prop('disabled', false);

           $("#response").html("Server load is high at the moment<br /><br />Please try again...")

           var debugInfo = "\nDEBUG INFO:\nClient timing (ms): " + time + "\n"
                                        + "Retries: "+this.tryCount+"\n"
                                        + "Error Reponse:\n"
                                        + JSON.stringify(data, null, "  ");
           console.log(debugInfo);
       },
     });
}
