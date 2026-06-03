from fastapi import FastAPI, Depends, HTTPException

from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from database import SessionLocal, engine, Base
import models
import os
import base64
import cv2
import numpy as np

from fastapi.responses import FileResponse
from fastapi import Form, UploadFile, File
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Creamos las tablas en la base de datos
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="API de Administración Electoral - Blindaje Biométrico",
    version="1.2.0"
)







# --- Dependencia: Conexión a la BD ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Esquemas Pydantic ---
class DNIRequest(BaseModel):
    dni: str

class HuellaRequest(BaseModel):
    votante_id: int
    huella_exitosa: bool

class RostroRequest(BaseModel):
    votante_id: int
    rostro_exitoso: bool
    foto_base64: str = None

class VotoRequest(BaseModel):
    votante_id: int
    partido_id: int

# --- NUEVO ESQUEMA ---
class PartidoCreate(BaseModel):
    nombre: str
    siglas: str
    foto_base64: str

# --- ENDPOINTS DE PARTIDOS ---
@app.post("/admin/partidos")
def crear_partido(request: PartidoCreate, db: Session = Depends(get_db)):
    try:
        # Decodificar imagen base64
        image_data = base64.b64decode(request.foto_base64)
        
        # Subir a Cloudinary
        upload_result = cloudinary.uploader.upload(
            image_data,
            folder="partidos",
            public_id=request.siglas.lower(),
            overwrite=True
        )
        foto_url = upload_result["secure_url"]

        # Evitar duplicados por siglas
        existente = db.query(models.PartidoPolitico).filter(
            models.PartidoPolitico.siglas == request.siglas
        ).first()
        if existente:
            raise HTTPException(status_code=400, detail=f"Las siglas '{request.siglas}' ya están registradas.")

        nuevo = models.PartidoPolitico(
            nombre=request.nombre,
            siglas=request.siglas,
            foto_url=foto_url
        )
        db.add(nuevo)
        db.commit()
        return {"mensaje": "Partido registrado con éxito"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
@app.delete("/admin/partidos/{partido_id}")
def eliminar_partido(partido_id: int, db: Session = Depends(get_db)):
    partido = db.query(models.PartidoPolitico).filter(
        models.PartidoPolitico.id == partido_id
    ).first()
    if not partido:
        raise HTTPException(status_code=404, detail="Partido no encontrado.")

    # Eliminar de Cloudinary
    try:
        public_id = f"partidos/{partido.siglas.lower()}"
        cloudinary.uploader.destroy(public_id)
    except Exception:
        pass  # Si falla Cloudinary igual eliminamos de BD

    db.delete(partido)
    db.commit()
    return {"mensaje": "Partido eliminado."}

@app.get("/partidos")
def listar_partidos(db: Session = Depends(get_db)):
    partidos = db.query(models.PartidoPolitico).all()
    # Devolvemos la lista para que el Votante la vea en su pantalla
    return [{"id": p.id, "nombre": p.nombre, "siglas": p.siglas, "foto_url": p.foto_url} for p in partidos]

@app.get("/admin/votantes")
def total_votantes(db: Session = Depends(get_db)):
    total = db.query(func.count(models.Votante.id)).scalar()
    return {"total": total}


 
@app.get("/admin")
def panel_admin():
    return FileResponse("admin_panel.html")


# --- Endpoints ---

@app.get("/")
def leer_raiz():
    return {"mensaje": "Servidor de Votación Activo - Motor Facenet512 Operativo"}

@app.post("/auth/register-dni")
def registrar_dni(request: DNIRequest, db: Session = Depends(get_db)):
    # 1. Validación básica del DNI
    if not request.dni.isdigit() or len(request.dni) != 8:
        raise HTTPException(status_code=400, detail="DNI inválido.")

    # 2. Buscar votante en BD y verificar que tiene embedding
    votante = db.query(models.Votante).filter(
        models.Votante.dni == request.dni
    ).first()

    if not votante or votante.face_embedding is None:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró registro biométrico para el DNI {request.dni}."
        )

    # 3. Bloquear doble voto
    if votante.ha_votado:
        raise HTTPException(
            status_code=403,
            detail="Este DNI ya emitió su voto."
        )

    # 4. Reset biométrico (nuevo intento de validación)
    votante.huella_validada = False
    votante.rostro_validado = False

    db.commit()
    db.refresh(votante)

    # 5. Respuesta al frontend
    return {
        "mensaje": "DNI reconocido correctamente",
        "votante_id": votante.id,
        "datos_oficiales": {
            "dni": request.dni,
            "foto_oficial_url": votante.foto_url or "",
            "nombre_simulado": "CIUDADANO REGISTRADO"
        }
    }

@app.get("/admin/verificaciones-faciales")
def listar_verificaciones(db: Session = Depends(get_db)):
    votantes = db.query(models.Votante).all()
    return [
        {
            "id": v.id,
            "dni": v.dni,
            "rostro_validado": v.rostro_validado,
            "huella_validada": v.huella_validada,
            "ha_votado": v.ha_votado,
            "foto_url": v.foto_url  # 👈 debe ser esto, no f"/static/{v.dni}.jpg"
        }
        for v in votantes
    ]

@app.post("/auth/verify-face")
def verificar_rostro(request: RostroRequest, db: Session = Depends(get_db)):

    import pickle

    votante = db.query(models.Votante).filter(
        models.Votante.id == request.votante_id
    ).first()

    if not votante:
        raise HTTPException(status_code=404, detail="Votante no encontrado.")

    if not request.foto_base64:
        raise HTTPException(status_code=400, detail="Captura facial vacía.")

    # 🔥 convertir imagen
    image_data = base64.b64decode(request.foto_base64)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        from deepface import DeepFace
        # 🔥 obtener embedding del rostro actual
        current = DeepFace.represent(
            img_path=img,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True
        )[0]["embedding"]

        # -------------------------
        # CASO 1: PRIMER REGISTRO
        # -------------------------
        if votante.face_embedding is None:

            votante.face_embedding = pickle.dumps(current)
            votante.rostro_validado = True
            db.commit()

            return {"mensaje": "Rostro registrado correctamente."}

        # -------------------------
        # CASO 2: COMPARACIÓN
        # -------------------------
        stored = pickle.loads(votante.face_embedding)

        distancia = np.linalg.norm(
            np.array(current) - np.array(stored)
        )

        match = distancia < 25 # umbral estable para embeddings

        if match:
            votante.rostro_validado = True
            db.commit()
            return {"mensaje": "Acceso biométrico concedido."}
        else:
            raise HTTPException(status_code=401, detail="Rostro no coincide.")

    except HTTPException:
        raise
    except Exception as e:
        print("ERROR BIOMETRÍA:", str(e))
        raise HTTPException(status_code=500, detail="Error en reconocimiento facial")
    
@app.post("/auth/verify-fingerprint")
def verificar_huella(request: HuellaRequest, db: Session = Depends(get_db)):
    votante = db.query(models.Votante).filter(models.Votante.id == request.votante_id).first()
    if not votante:
        raise HTTPException(status_code=404, detail="Votante no encontrado.")
    
    if request.huella_exitosa:
        votante.huella_validada = True
        db.commit()
        return {"mensaje": "Huella validada con éxito."}
    else:
        raise HTTPException(status_code=401, detail="Fallo en validación de huella.")

@app.post("/voting/cast")
def emitir_voto(request: VotoRequest, db: Session = Depends(get_db)):
    votante = db.query(models.Votante).filter(models.Votante.id == request.votante_id).first()
    
    if votante.ha_votado:
        raise HTTPException(status_code=403, detail="Ya has votado.")
    
    if not votante.huella_validada or not votante.rostro_validado:
        raise HTTPException(status_code=403, detail="Falta validación multifactor.")
        
    nuevo_voto = models.Voto(partido_id=request.partido_id)
    votante.ha_votado = True 
    db.add(nuevo_voto)
    db.commit()
    return {"mensaje": "Voto registrado correctamente."}

@app.get("/admin/results")
def conteo_de_votos(db: Session = Depends(get_db)):
    resultados = db.query(
        models.PartidoPolitico.nombre,
        models.PartidoPolitico.siglas,
        func.count(models.Voto.id).label("total_votos")
    ).outerjoin(
        models.Voto, models.PartidoPolitico.id == models.Voto.partido_id
    ).group_by(
        models.PartidoPolitico.id
    ).all()
    
    reporte = [{"partido": n, "siglas": s, "votos": t} for n, s, t in resultados]
    return {"mensaje": "Reporte de resultados", "resultados": reporte}

@app.post("/admin/upload-dni-foto")
def subir_foto_dni(
    dni: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not dni.isdigit() or len(dni) != 8:
        raise HTTPException(status_code=400, detail="DNI inválido")

    image_data = file.file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Imagen inválida")

    try:
        import pickle
        from deepface import DeepFace
        # Subir foto a Cloudinary
        upload_result = cloudinary.uploader.upload(
            image_data,
            folder="dni_fotos",
            public_id=dni,
            overwrite=True
        )
        foto_url = upload_result["secure_url"]

        # Generar embedding facial
        embedding = DeepFace.represent(
            img_path=img,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True
        )[0]["embedding"]

        # Buscar o crear votante
        votante = db.query(models.Votante).filter(
            models.Votante.dni == dni
        ).first()

        if not votante:
            votante = models.Votante(
                dni=dni,
                huella_validada=False,
                rostro_validado=False,
                ha_votado=False,
                face_embedding=pickle.dumps(embedding),
                foto_url=foto_url
            )
            db.add(votante)
        else:
            votante.face_embedding = pickle.dumps(embedding)
            votante.foto_url = foto_url

        db.commit()
        return {"mensaje": f"Foto del DNI {dni} procesada correctamente", "foto_url": foto_url}

    except Exception as e:
        print("ERROR DETALLADO:", str(e))
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")