# article-writer

Escritor automático de artículos de ciencias sociales en el estilo de
[Revista La Cueva](https://revistalacueva.wordpress.com/). Va del tema a la versión
final revisada, con bibliografía real y verificada, y pasada por detectores de IA.

## Instalación

```bash
pip install -r requirements.txt
cp .env.example .env             # y poné tu clave en AW_API_KEY
python main.py --setup           # tus intereses, el corpus de referencia y la guía de estilo
python ingest_facultad.py        # opcional: suma tus trabajos de facultad al corpus
python main.py --refresh-style   # y reconstruí la guía con el corpus ampliado
```

`--setup` primero te muestra tus intereses (ver abajo) y te deja reescribirlos; después
descarga los 16 artículos de referencia a `corpus/`, mide la huella
estilométrica del autor (`style_fingerprint.json`) y hace que el modelo PRO escriba
la guía de estilo operativa (`style_guide.md`). Se corre una sola vez.

`ingest_facultad.py` recorre `D:/facultad` (o la ruta que le pases) y suma tus
monografías, parciales domiciliarios e informes a `corpus_facultad/`. Filtra en tres
pasos: extensión (vos escribís en Word, las lecturas son PDF), nombre de archivo
(`Antropología 24-9.docx` es un apunte fechado, no un ensayo) y contenido — FLASH lee
una muestra y decide si sos vos argumentando o si es texto ajeno, apuntes o una
desgrabación. Con `--dry-run` ves la clasificación sin escribir nada.

`corpus_facultad/` es **solo referencia de estilo**: nunca entra en la bibliografía ni
se cita. La bibliografía sale de los conectores online y de `library/`, que son
directorios distintos a propósito.

### Intereses

Los temas salen de `interests.txt`: un área por línea, las líneas con `#` se ignoran.
Es personal y no se sube al repositorio; un clon nuevo no lo tiene, así que el primer
`--setup` copia `interests.example.txt` (la lista del autor, como ejemplo) y te
pregunta si querés reemplazarla. Escribís un área por línea y una línea vacía para
terminar; Enter directo la deja como está. También podés editar el archivo a mano en
cualquier momento, y `--setup` vuelve a ofrecerlo cada vez que lo corras.

Cada ronda de temas sortea cuatro áreas de esa lista, busca novedades de esas áreas
(el primer término de cada línea es la consulta) y le pide a PRO cinco temas de áreas
distintas entre sí, sin repetir el asunto de ningún artículo ya escrito (los doce más
recientes van aparte, así nunca se pierden por recorte). La economía política puede ser
una lente, no el eje de todo: un artículo sobre dinosaurios o sobre aviones vale por sí
mismo.

## Uso

```bash
python main.py                                  # asistente interactivo
python main.py --fmt medium --mode auto          # de punta a punta, sin preguntar
python main.py --topic "Cybersyn hoy" --exact-topic --fmt long
python main.py --topic "ciencia ficción y valor" --mode assisted
python main.py --resume                         # retomar la última corrida (cada etapa queda cacheada)
python main.py --resume output/20260820-slug    # retomar una corrida puntual
python main.py --detect texto.md                # solo pasar los detectores de IA
python main.py --spanish --fmt medium            # artículo en español (el inglés es el default)
```

### Banderas

| Bandera | Qué hace |
|---|---|
| `--fmt` | extensión y género (tabla de abajo) |
| `--mode auto\|assisted` | de punta a punta, o con consultas en tema, esquema, correcciones y aprobación |
| `--topic`, `--exact-topic` | pista de tema; con `--exact-topic`, ese tema y ningún otro |
| `--resume [carpeta]` | retoma una corrida; sin carpeta, la última |
| `--no-library` | no frena a pedir libros faltantes |
| `--rounds N` | rondas de revisión de PRO |
| `--detector-rounds N`, `--threshold N`, `--no-detector` | reescrituras contra los detectores, puntaje máximo tolerado (0 humano, 100 máquina), o sin detectores |
| `--drafter pro\|flash` | quién redacta las secciones (FLASH por defecto; PRO lee más humano y cuesta más) |
| `--english`, `--spanish` | idioma del artículo; inglés por defecto, calibrado contra `corpus_en/` |
| `--setup`, `--refresh-style` | intereses + corpus + guía de estilo; o solo reconstruir la guía |
| `--detect archivo.md` | solo los detectores sobre un archivo |
| `--publish no\|draft\|auto\|live` | subir como borrador, en vivo si aprueba y pasa el umbral, o siempre en vivo |
| `--provider`, `--backups A,B`, `--pro`, `--flash`, `--models` | cadena de proveedores y modelos (ver Modelos) |
| `--wizard` | el asistente pregunta aunque haya otras banderas |
| `--continuous N`, `--every MIN` | N artículos seguidos eligiendo temas solo (0 = sin fin), con pausa entre uno y otro |

**Vos ponés el tema.** Con `--topic` das una pista y el sistema te propone cinco ángulos
para elegir. Con `--topic --exact-topic` escribe sobre eso y nada más: PRO solo le agrega
hipótesis, pregunta y plan bibliográfico. En modo asistido podés ajustar la hipótesis
antes de que empiece a investigar.

### Formatos

| `--fmt` | Palabras | Qué es |
|---|---|---|
| `short` | 1.200–1.600 | artículo breve, polémico, de intervención |
| `medium` | 3.000–4.000 | artículo de divulgación teórica |
| `long` | 7.000–9.000 | ensayo extenso con aparato crítico |
| `paper` | 10.000–13.000 | paper académico con resumen, hipótesis y conclusión |
| `thesis` | 40.000–80.000 | tesis por capítulos |
| `book` | 70.000–120.000 | libro de ensayo teórico |
| `debate` | 3.500–4.500 | confrontación explícita entre posiciones rivales |

La extensión es un **rango, no una meta**: el esquema reparte el presupuesto entre las
secciones y el redactor decide cuántas y de qué largo (500 a 10.000 palabras, con
longitudes explícitamente desiguales). Cada formato lleva además su propio registro
(un `paper` y un artículo de intervención no hablan igual), que manda sobre las reglas
de estilo genéricas.

### Modos

- `auto` — elige tema, investiga, escribe, revisa y aprueba sin intervención. No
  muestra la lista de temas propuestos y toma uno al azar (siempre el primero era
  siempre la idea más previsible del modelo).
- `asistido` — te consulta en cuatro puntos: elección del tema, esquema, aplicación
  de las correcciones de la revisión, y aprobación final.

## Cómo funciona

```
tema (PRO)  →  investigación  →  esquema (FLASH, auditado por PRO)
            →  redacción por secciones (FLASH)
            →  detectores de IA + reescritura (FLASH + jueces de otras familias)
            →  revisión de pares (PRO)  →  corrección (FLASH)
            →  aprobación final (PRO)
```

Cada etapa se guarda en `output/<fecha>-<slug>/`:

| Archivo | Contenido |
|---|---|
| `01_topic.json` | tema, hipótesis, tensión teórica, por qué ahora |
| `02_plan.json` / `02_dossier.json` / `02_faltantes.json` | consultas, fuentes, libros que no se consiguieron |
| `03_plan.json` | arquitectura previa (sólo `tesis` y `libro`): unidades, función y presupuesto |
| `03_outline.json` | esquema con presupuesto de palabras y fuentes por sección |
| `04_sec01.md`… | cada sección redactada (caché: borrar para rehacerla). Las secciones largas se parten por subsección: `04_sec02_01.md`, y en partes si aún no entran: `04_sec02_01a.md` |
| `04_draft.md`, `04_draft_humanizado.md` | borrador y versión desautomatizada |
| `05_final.md` | **el artículo** |
| `06_review.json`, `07_detector.json`, `08_approval.json` | revisión, detectores, dictamen |

## Bibliografía

Fuentes consultadas, todas sin clave de API:

- **Papers**: OpenAlex, Crossref, DOAJ (buena cobertura en español), Semantic Scholar, arXiv.
- **Texto completo**: Unpaywall (acceso abierto legal) y, para lo que está tras muro
  de pago, Sci-Hub. Los espejos de Sci-Hub están casi siempre detrás de un desafío JS,
  así que suele fallar; no es un error del programa. Cuando ningún catálogo resuelve
  una obra, la búsqueda termina en la web abierta con Chrome sin cabeza (Selenium), la
  única ruta que sigue pasando donde los buscadores bloquean el HTTP sin clave.
- **Actualidad**: Google News RSS (español y inglés) y GDELT.
- **Libros**: primero tu biblioteca Calibre (consultada por título, sin escanearla),
  después Open Library, Project Gutenberg, archive.org y Library Genesis; lo que
  aterriza se guarda en `library/` para la próxima corrida. Artículos de revista los
  resuelve `fetch_paper()` (Crossref/arXiv → DOI → Unpaywall → Sci-Hub), y Anna's
  Archive queda solo como lista de enlaces — Cloudflare la bloquea.
- **Archivos de teoría**: Monthly Review, Viewpoint, Brooklyn Rail, Spectre,
  Historical Materialism, libcom.org, y la propia Revista La Cueva.
- **Tu biblioteca**: poné `.epub`, `.pdf`, `.docx`, `.txt`, `.html` (o formatos
  Calibre como `.azw3`, que se convierten con `ebook-convert`) en `library/` y se
  indexan solos.

### El portón de biblioteca

Antes de escribir una sola línea, el sistema compara las obras que el plan declaró
imprescindibles contra las que efectivamente consiguió **en texto completo** (un resumen
de dos líneas no cuenta). Con lo que falte, frena y te muestra, obra por obra, los
candidatos de descarga directa que encontró más siete enlaces de respaldo —Anna's
Archive, Library Genesis, Z-Library, Marxists Internet Archive, Google Books, Internet
Archive, Open Library—. Dejás los archivos en `library/`, apretás Enter, y vuelve a
chequear; también podés pegar una URL y la baja en el momento. Hasta tres rondas.
Escribí `skip` para continuar sin ellas. En modo `auto` imprime el informe y sigue
solo, sin bloquear. Con `--no-library` ni pregunta.

### Citas: APA 7 y nada inventado

Se cita en **normas APA 7 en español**: `(Postone, 2006)`, `(Postone, 2006, p. 302)`,
`(Kurz y Jappe, 2016)`, `(Marx et al., 1867)`, cita narrativa cuando el autor es sujeto,
y lista de **Referencias** alfabética con títulos en cursiva y DOI.

**Ninguna cita se inventa.** El modelo solo puede citar claves presentes en el dossier;
`research.verify_citations()` cruza cada `(Autor, año)` del texto contra las fuentes
reales, antes y después de la reescritura, y manda a corregir lo que no tenga respaldo.

## Revisión de prosa y detectores

La redacción y la reescritura comparten criterios editoriales en inglés y español:
claridad, continuidad del argumento, precisión y sintaxis idiomática. Las muestras
del corpus orientan la voz; sus datos y citas no son fuentes del artículo. Las
métricas describen el estilo y no imponen cuotas de frases cortas, subordinadas,
paréntesis, enclisis ni signos de puntuación.

Cada sección nueva se comprueba antes de guardarla: extensión dentro del rango
pedido e idioma compatible. Una respuesta inválida se vuelve a pedir una vez;
si vuelve a fallar, la ejecución se detiene sin guardarla como una sección terminada.
La revisión y la corrección reciben los pasajes disponibles del dossier para poder
contrastar las afirmaciones con las fuentes.

La reescritura recibe el contexto de los bloques vecinos. Cada propuesta debe
conservar las citas, páginas, cifras, citas textuales y marcas de notas, y mantenerse
dentro de ±10% de la extensión original. Si altera esos elementos, se conserva el
bloque anterior. Esta comprobación no verifica por sí sola el significado: la
fidelidad de las afirmaciones sigue requiriendo revisión editorial y documental.

El informe separa tres señales:

1. Estilometría local: diagnósticos para revisar la prosa, sin decidir la aprobación.
2. Jueces LLM: los modelos PRO y FLASH elegidos para la ejecución, sobre el inicio,
   centro y final de textos largos. No son una evaluación independiente del redactor.
3. Detectores externos: los servicios configurados y, en inglés, el clasificador
   local opcional si está disponible.

El umbral se aplica a la peor puntuación válida de los jueces y detectores externos.
Sin respuestas válidas, el resultado es indeterminado, nunca aprobado. Se conserva
la mejor versión medida y el informe identifica su ronda, idioma y huella SHA-256.
Si una corrección posterior cambia el texto, se vuelve a medir. La publicación
automática exige aprobación editorial y un resultado explícito por debajo del umbral;
omitir los detectores no cuenta como haberlos pasado.

**Inglés (default; `--spanish` para español).** Usa `corpus_en/`, la guía inglesa y criterios de
redacción propios del idioma; también traduce los rótulos, la fecha y la bibliografía
generados por el programa. `python build_corpus_en.py --guide` prepara ese corpus.
**Consola y prompts en inglés.** Todos los mensajes de la consola, el asistente y las
preguntas interactivas (`[y/N]`, `[e]dit / [p]ublish / [n]o`, `[w]ait / [c]hange / [a]bort`)
están en inglés, y también las instrucciones que reciben los modelos; el idioma del
artículo lo fija una directiva al final de cada prompt (`pipeline._lang()`), así que
`--spanish` sigue escribiendo en español rioplatense. Los nombres de campos JSON y
los archivos de `output/` no cambian, así que los runs viejos se retoman igual.
El juez y el reescritor específicos del español (`humanize.JUDGE_PROMPT`,
`REWRITE_PROMPT`) y la construcción de la guía española (`style.BUILD_PROMPT`)
siguen en español: solo corren con `--spanish`.
Sin corpus o guía ingleses, puede escribir con los criterios generales. Las llamadas
a los escritores CLI se aíslan de las instrucciones de programación del proyecto.

Los puntajes son señales de revisión, no pruebas de autoría. Una puntuación baja
no garantiza que otro detector acepte el texto ni que sus afirmaciones sean correctas.

## Modelos

| Rol | Modelo | Para qué |
|---|---|---|
| PRO | el del proveedor principal | tema, auditoría del esquema, revisión, correcciones, aprobación final |
| FLASH | el del proveedor principal | consultas, esquema, redacción, reescritura |
| Jueces | PRO y FLASH | detección de texto generado |

Cada respaldo de la cadena contesta con sus propios modelos (`--models`, `AW_MODELS`):

| Proveedor | PRO / FLASH por defecto |
|---|---|
| `claude` | `claude-opus-5-5` / `sonnet` |
| `hyper`, `go` | `qwen3.8-flash` / `deepseek-v4.1-flash` |
| `zen` | `glm-5.3-flash` / `deepseek-v4.1-flash` |
| `grok` | `grok-4` / `grok-4-fast` |
| `oauth` | `gpt-6-sol` / `gpt-6-luna` |

Se cambian por `.env` (`AW_BACKEND`, `AW_MODEL_PRO`, `AW_MODEL_FLASH`,
`AW_MODEL_JUDGES`), por banderas (`--provider/--backups/--pro/--flash`) o en el
asistente interactivo (primera pregunta). `AW_BACKEND` es una **cadena ordenada** de
proveedores: `hyper` por defecto, `claude` y `opencode` también disponibles; la
llamada sale por el primero y camina al siguiente si no responde. `claude` es el CLI
de Claude Code en modo print (corre sobre tu suscripción, no sobre una clave medida)
y es *exclusivo*: solo responde modelos de su catálogo, así que los jueces nunca
caen en la familia que escribió el texto.

### Respaldo por OpenCode

Si el proveedor principal no responde después de todos los reintentos, la llamada sale
por el CLI de `opencode` contra el provider `opencode-go`, que sirve las mismas
familias (y alguna que hyper no tiene, como `glm-5.3`). Es automático; se apaga con
`AW_OPENCODE_FALLBACK=0`. En los menús y en `AW_BACKEND` el proveedor se llama `go`
(el nombre viejo `opencode` se sigue aceptando).

Todas las llamadas salen por el `AIService` de `book writer` (la suite de IA común del
workspace; carpeta hermana `../book writer` o `AW_BOOK_WRITER`): cada eslabón de la
cadena usa la configuración de ese proveedor en book writer, pide el máximo de salida
del modelo, reintenta una respuesta truncada con el doble de presupuesto, y el último
eslabón de una corrida desatendida espera a que se libere un límite de uso en vez de
morir.

El proyecto trae un `opencode.json` con el provider `hyper` ya configurado y con
`CLAUDE.md` como instrucciones, por si querés trabajar el repo desde ahí:

```bash
export AW_API_KEY=sk-hyper-...
opencode        # desde article-writer/
```

## Lamplight

El juego (biblioteca, campaña histórica y cliente Godot) se separó el 2026-09-23 a
[`../lamplight`](../lamplight/README.md). Importa `llm`, `pipeline`, `research` y
`style` desde esta carpeta: si cambia su API pública, corré también sus pruebas.

## Verificación

```bash
python test_article_writer.py    # comprobaciones deterministas, sin red ni clave de API
python -m pytest -q -p no:cacheprovider   # alternativa, si tenés pytest instalado
python bench_bilingual.py --backend hyper  # prueba real, consume la API configurada
```

La prueba bilingüe guarda borradores, versiones editadas e informes en
`output/bilingual-check-<fecha>/`. Usa un caso explícitamente ficticio para comprobar
idioma, extensión del borrador, citas y preservación del contenido durante la
reescritura. La revisión puede acortar material repetido o sin respaldo; la reescritura
de estilo debe respetar ±10% de la versión revisada. Un editor LLM evalúa naturalidad,
claridad, coherencia, precisión y fidelidad; exige al menos 4/5 en cada criterio.
Los resultados del detector se informan por separado y no se presentan como pruebas
de calidad ni de autoría humana.
