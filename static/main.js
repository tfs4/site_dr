
const uploadForm = document.getElementById('upload-form')
const input = document.getElementById('id_image')
console.log(input)

const alertBox = document.getElementById('alert-box')
const imageBox = document.getElementById('image-box')
const progressBox = document.getElementById('progress-box')
const cancelBox = document.getElementById('cancel-box')
const cancelBtn = document.getElementById('cancel-btn')

const csrf = document.getElementsByName('csrfmiddlewaretoken')

input.addEventListener('change', ()=>{
    progressBox.classList.remove('not-visible')
    cancelBox.classList.remove('not-visible')
    const img_data = input.files[0]
    const url = URL.createObjectURL(img_data)
    console.log(img_data)


    const fd = new FormData()
    fd.append('csrfmiddlewaretoken', csrf[0].value)
    fd.append('image', img_data)

    $.ajax({
        type:'POST',
        url: uploadForm.action,
        enctype: 'multipart/form-data',
        data: fd,
        beforeSend: function(){
            console.log('before')
            alertBox.innerHTML= ""
            imageBox.innerHTML = ""
        },
        xhr: function(){
            const xhr = new window.XMLHttpRequest();
            /*xhr.upload.addEventListener('progress', e=>{*/
            xhr.upload.addEventListener('progress', e=>{
                // console.log(e)
                if (e.lengthComputable) {
                    const percent = e.loaded / e.total * 100
                    console.log(percent)
                    progressBox.innerHTML = ` 
                                                <div class="progress">
                                               
                                                <div class="progress-bar" role="progressbar" style="width: ${percent}%" aria-valuenow="${percent}" aria-valuemin="0" aria-valuemax="100"></div>
                                            </div>
                                            <p>${percent.toFixed(1)}%</p>`
                }

            })
            /**/
            cancelBtn.addEventListener('click', ()=>{
                xhr.abort()
                setTimeout(()=>{
                    uploadForm.reset()
                    progressBox.innerHTML=""
                    alertBox.innerHTML = ""
                    cancelBox.classList.add('not-visible')
                }, 2000)
            })

            imageBox.innerHTML = `<div id="load_process"><br /><br /><br /><br /><br /><br /><p>Analyzing the image</p><div id="loading"></div></div>`
            return xhr
        },
        success: function(response){
            console.log(response)


            progressBox.innerHTML = ``
            imageBox.innerHTML = ``
            alertBox.innerHTML = `
<div class="alert alert-success" role="alert"> Result:  ${response.message}</div>
 <img src="${url}" id="image_upada">


                                <div id="classificacao_img" class="classificacao_img_cls">
                                        <ul>
                                                    <li class="${response.item_0}"><b>Stage 0</b>: No diabetic retinopathy.</li>
                                                    <li class="${response.item_1}"><b>Stage 1</b>: Mild nonproliferative diabetic retinopathy.</li>
                                                    <li class="${response.item_2}"><b>Stage 2</b>: Moderate nonproliferative diabetic retinopathy.</li>
                                                    <li class="${response.item_3}"><b>Stage 3</b>: Severe nonproliferative diabetic retinopathy.</li>
                                                    <li class="${response.item_4}"><b>Stage 4</b>: Proliferative diabetic retinopathy.</li>
                                         </ul>
                                    </div>`



            cancelBox.classList.add('not-visible')
        },
        error: function(error){
            console.log(error)
            alertBox.innerHTML = `<div class="alert alert-danger" role="alert">
                                    Something went wrong
                                </div>`
        },
        cache: false,
        contentType: false,
        processData: false,
    })
})
function classifie_one(type){
    let url = "{% url 'quantities' %}".replace('ok', type);
    $.ajax({
      method: 'POST',
      url: url,
      data: {
          csrfmiddlewaretoken: '{{ csrf_token }}'
      }
  });
  }